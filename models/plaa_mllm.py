
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
        self.llm_infer = LLMInference(model_config, device_config)

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
                pass
        
        vision_tokens = self.forensic_cross_attention(
            semantic_features,
            artifact_features,
            text_guidance_tensor
        )
        
        projected_vision_tokens = self.vision_token_proj(vision_tokens)
        
        pooled_vision = projected_vision_tokens.mean(dim=1)
        detection_logits = self.detection_head(pooled_vision)
        
        pred_mask = None
        try:
            mask_flat = self.mask_head(pooled_vision)
            pred_mask = mask_flat.view(-1, 1, 224, 224)
        except:
            pass
        
        llm_outputs = {}
        try:
            if self.llm_infer.tokenizer is not None:
                tokenized = self.llm_infer.tokenizer(
                    text_prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.model_config.max_seq_len
                ).to(image.device)
                
                inputs_embeds, fused_attention_mask = self._early_fusion(
                    projected_vision_tokens,
                    tokenized
                )
                
                llm_out = self.llm_infer.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=fused_attention_mask
                )
                llm_outputs['logits'] = llm_out.logits
        except Exception as e:
            pass
        
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
            
            detection_logits = outputs.get('detection_logits')
            confidence = torch.sigmoid(detection_logits).item() if detection_logits is not None else 0.5
            
            try:
                explanation = self.llm_infer.generate_explanation(image, "Explain your classification decision.")
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
            self.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        except Exception as e:
            pass

    def save_checkpoint(self, checkpoint_path):
        try:
            torch.save({'model_state_dict': self.state_dict()}, checkpoint_path)
        except Exception as e:
            pass

