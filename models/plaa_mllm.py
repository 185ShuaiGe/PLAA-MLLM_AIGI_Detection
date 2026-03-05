
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.dual_stream_encoder import DualStreamEncoder
from models.forensic_cross_attention import ForensicCrossAttention
from models.llm_infer import LLMInference


class PLAAMLLM(nn.Module):
    def __init__(self, model_config, device_config, path_config):
        super().__init__()
        self.model_config = model_config
        self.device_config = device_config
        self.path_config = path_config

        self.dual_stream_encoder = DualStreamEncoder(model_config, device_config)
        self.forensic_cross_attention = ForensicCrossAttention(model_config, device_config)
        self.llm_infer = LLMInference(model_config, device_config)          #日志第一条信息：加载大语言模型

        self.vision_token_proj = nn.Linear(model_config.latent_dim, model_config.llm_dim)
        
        self.detection_head = nn.Sequential(
            nn.Linear(model_config.llm_dim, model_config.llm_dim // 2),
            nn.GELU(),
            nn.Linear(model_config.llm_dim // 2, 1)
        )
        
        self.mask_head = nn.Sequential(
            nn.Linear(model_config.llm_dim, model_config.latent_dim),
            nn.GELU(),
            nn.Linear(model_config.latent_dim, 224 * 224)
        )

    def forward(self, image, text_prompt, text_guidance=None):
        # 自动判断视觉编码器是否被冻结，若冻结则不记录梯度图，省下海量中间激活值的显存
        # visual_requires_grad = any(p.requires_grad for p in self.dual_stream_encoder.parameters())
        # with torch.set_grad_enabled(visual_requires_grad):
        semantic_features, artifact_features = self.dual_stream_encoder(image)
        
        text_guidance_tensor = None
        if text_guidance is not None and self.llm_infer.tokenizer is not None:
            try:
                tokenized = self.llm_infer.tokenizer(
                    text_guidance,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(image.device)
                text_guidance_tensor = self.llm_infer.llm_model.get_input_embeddings()(tokenized['input_ids'])
            except:
                raise ValueError("Failed to tokenize text guidance. Please check the input and tokenizer configuration.")
        
        vision_tokens = self.forensic_cross_attention(
            semantic_features,
            artifact_features,
            text_guidance_tensor
        )
        
        projected_vision_tokens = self.vision_token_proj(vision_tokens)
        
        pooled_vision = projected_vision_tokens.mean(dim=1)
        detection_logits = self.detection_head(pooled_vision)
        
        # （plaa_mllm.py forward 函数的后半部分）
        pred_mask = None
        try:
            mask_flat = self.mask_head(pooled_vision)
            pred_mask = mask_flat.view(-1, 1, 224, 224)
        except:
            pass
        
        llm_outputs = {}
        # ==================== 【核心修复 2】 ====================
        # Stage 1 时 LLM 没被解冻，强行执行不仅没用，还会严重拖慢整批训练的速度！
        llm_is_frozen = not any(p.requires_grad for p in self.llm_infer.parameters())
        if self.training and llm_is_frozen:
            skip_llm = True
        else:
            skip_llm = False
            
        if not skip_llm:
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
        # =========================================================
        
        return {
            'vision_tokens': projected_vision_tokens,
            'detection_logits': detection_logits,
            'pred_mask': pred_mask,
            **llm_outputs
        }

    def detect_image(self, image, text_guidance=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image, "Classify this image as real or AI-generated.", text_guidance)
            
            projected_vision_tokens = outputs.get('vision_tokens')
            detection_logits = outputs.get('detection_logits')
            confidence = torch.sigmoid(detection_logits).item() if detection_logits is not None else 0.5
            
            try:
                explanation = self.llm_infer.generate_explanation(
                    image_features=projected_vision_tokens, 
                    detection_score=confidence, 
                    prompt=text_guidance
                )
            except:
                explanation = "AI-generated" if confidence > 0.5 else "Real"
        
        return confidence, explanation

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
                
                # 2. 过滤掉 LLM 冻结的主干权重，但必须保留 LoRA 权重
                if clean_k.startswith('llm_infer.llm_model.') and 'lora' not in clean_k:
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
            print(f"Error loading checkpoint in PLAAMLLM: {e}")

    def save_checkpoint(self, checkpoint_path):
        try:
            torch.save({'model_state_dict': self.state_dict()}, checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

