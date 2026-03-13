
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.dual_stream_encoder import DualStreamEncoder
from models.mome_fusion import MoMEFusion
from models.llm_infer import LLMInference


class DSMoME(nn.Module):
    def __init__(self, model_config, device_config, path_config):
        super().__init__()
        self.model_config = model_config
        self.device_config = device_config
        self.path_config = path_config

        self.dual_stream_encoder = DualStreamEncoder(model_config, device_config)
        self.mome_fusion = MoMEFusion(model_config, device_config)
        self.llm_infer = LLMInference(model_config, device_config)          #日志第一条信息：加载大语言模型

        self.vision_token_proj = nn.Linear(model_config.latent_dim, model_config.llm_dim)
        
        self.detection_head = nn.Sequential(
            nn.Linear(model_config.llm_dim, model_config.llm_dim // 2),
            nn.GELU(),
            nn.Linear(model_config.llm_dim // 2, 1)
        )

    def forward(self, image, text_prompt, text_guidance=None):
        # 自动判断视觉编码器是否被冻结，若冻结则不记录梯度图，省下海量中间激活值的显存
        visual_requires_grad = any(p.requires_grad for p in self.dual_stream_encoder.parameters())
        
        semantic_features = None
        artifact_features = None
        vision_tokens = None
        projected_vision_tokens = None
        pooled_vision = None
        detection_logits = None
        
        with torch.set_grad_enabled(visual_requires_grad):
            semantic_features, artifact_features = self.dual_stream_encoder(image)
            
            # MoME 融合不需要文本引导
            vision_tokens = self.mome_fusion(
                semantic_features,
                artifact_features
            )
            
            projected_vision_tokens = self.vision_token_proj(vision_tokens)
            
            pooled_vision = projected_vision_tokens.mean(dim=1)
            detection_logits = self.detection_head(pooled_vision)
        
        llm_outputs = {}
        # 即使 LLM 被冻结，也必须执行前向传播以实现特征对齐
        try:
            if self.llm_infer.tokenizer is not None:
                tokenized = self.llm_infer.tokenizer(
                    text_prompt,
                    return_tensors='pt',
                    padding=True,          # 必须加 padding=True 才能支持输入 text list！
                    truncation=True,
                    max_length=self.model_config.max_seq_len
                ).to(image.device)
                
                llm_out = self.llm_infer.forward(
                    input_ids=tokenized['input_ids'],
                    attention_mask=tokenized['attention_mask'],
                    vision_tokens=projected_vision_tokens
                )
                llm_outputs.update(llm_out)
        except Exception as e:
            print(f"Error in LLM forward pass: {e}")
        
        return {
            'vision_tokens': projected_vision_tokens,
            'detection_logits': detection_logits,
            **llm_outputs
        }

    def detect_image(self, image, text_guidance=None):
        self.eval()
        with torch.no_grad():
            # ======== 【核心修复 1】：加入与训练时完全相同的 bfloat16 混合精度 ========
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.forward(image, "<image>\nAnalyze this image and determine if it is real or AI-generated.", text_guidance)
                
                detection_logits = outputs.get('detection_logits')
                confidence = torch.sigmoid(detection_logits).item() if detection_logits is not None else 0.5
            # =====================================================================
        
        return confidence

    def _early_fusion(self, vision_tokens, text_tokens):
        input_ids = text_tokens['input_ids']
        attention_mask = text_tokens['attention_mask']
        
        try:
            text_embeds = self.llm_infer.llm_model.get_input_embeddings()(input_ids)
        except:
            text_embeds = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                self.model_config.llm_dim,
                device=input_ids.device
            )
        
        inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
        
        vision_attention_mask = torch.ones(
            vision_tokens.shape[0],
            vision_tokens.shape[1],
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        fused_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)
        
        return inputs_embeds, fused_attention_mask

    def load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device_config.get_device())
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # 1. 自动剥离多卡训练 (DDP) 产生的 module. 前缀，或 torch.compile 产生的 _orig_mod. 前缀
                clean_k = k.replace('module.', '').replace('_orig_mod.', '')
                
                # 2. 过滤掉 LLM 冻结的主干权重
                if clean_k.startswith('llm_infer.llm_model.'):
                    continue
                    
                filtered_state_dict[clean_k] = v
            
            # 3. 加载过滤并清理前缀后的权重，并捕获返回值
            res = self.load_state_dict(filtered_state_dict, strict=False)
            print(f"[{checkpoint_path}] Checkpoint successfully loaded.")
            
            # ================= 核心伤情诊断：检查视觉流是否加载成功 =================
            vision_missing = [k for k in res.missing_keys if 'dual_stream' in k or 'detection_head' in k]
            if vision_missing:
                print(f"🚨 严重警告: 共有 {len(vision_missing)} 个视觉/分类参数未能对齐加载！")
                print(f"🚨 示例缺失项: {vision_missing[:3]}")
            else:
                print("✅ 视觉编码器 (Dual Stream) 和分类头 (Detection Head) 权重已完美挂载！")
            # =======================================================================
            
        except Exception as e:
            print(f"Error loading checkpoint in DSMoME: {e}")

    def save_checkpoint(self, checkpoint_path):
        try:
            torch.save({'model_state_dict': self.state_dict()}, checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

