
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig


class LLMInference(nn.Module):
    def __init__(self, config: ModelConfig, device_config: DeviceConfig):
        super().__init__()
        self.config = config
        self.device_config = device_config
        self.llm_model = None
        self.tokenizer = None
        self.lora_config = None
        
        self.device = device_config.get_device()
        self._init_tokenizer()
    
    def _init_tokenizer(self) -> None:
        """
        初始化 Tokenizer
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None
    ):
        """
        LLM 前向传播

        Args:
            input_ids: 文本输入 token ID，形状 [B, T]，其中 B=batch_size, T=token长度
            attention_mask: 注意力掩码，形状 [B, T]
            vision_tokens: 可选的视觉令牌，形状 [B, N, D]，来自 ForensicCrossAttention

        Returns:
            Dict[str, torch.Tensor]: LLM 输出字典
                - 'logits': 输出 logits，形状 [B, N+T, V]，V=vocab_size
                    注意：序列长度为 N（视觉令牌数量）+ T（文本令牌数量）
                - 'loss': 可选的损失值（训练时）
        """
        if self.llm_model is None:
            return {}
        
        inputs_embeds = None
        fused_attention_mask = attention_mask
        
        if vision_tokens is not None:
            if hasattr(self.llm_model, 'get_input_embeddings'):
                text_embeds = self.llm_model.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
                
                vision_mask = torch.ones(
                    (vision_tokens.size(0), vision_tokens.size(1)), 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                fused_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        if inputs_embeds is not None:
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=fused_attention_mask
            )
        else:
            outputs = self.llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        return {'logits': outputs.logits}
    
    def generate(
        self,
        prompt: str,
        vision_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256
    ) -> str:
        """
        生成自然语言检测结果和解释

        Args:
            prompt: 输入文本提示
            vision_tokens: 可选的视觉令牌，形状 [B, N, D]
            max_new_tokens: 最大生成长度

        Returns:
            str: 生成的自然语言检测结果
        """
        if self.tokenizer is None or self.llm_model is None:
            return self._fallback_explanation(0.5)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            inputs_embeds = None
            if vision_tokens is not None:
                if hasattr(self.llm_model, 'get_input_embeddings'):
                    text_embeds = self.llm_model.get_input_embeddings()(input_ids)
                    inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
                    
                    vision_mask = torch.ones((vision_tokens.size(0), vision_tokens.size(1)), 
                                            dtype=torch.long, device=self.device)
                    fused_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
                    attention_mask = fused_attention_mask
            
            generate_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': getattr(self.config, 'temperature', 0.7),
                'top_p': getattr(self.config, 'top_p', 0.9),
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }
            
            if inputs_embeds is not None:
                generate_kwargs['inputs_embeds'] = inputs_embeds
                generate_kwargs['attention_mask'] = attention_mask
            else:
                generate_kwargs['input_ids'] = input_ids
                generate_kwargs['attention_mask'] = attention_mask
            
            with torch.no_grad():
                output_ids = self.llm_model.generate(**generate_kwargs)
            
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return self._postprocess_explanation(generated_text)
            
        except Exception as e:
            return self._fallback_explanation(0.5)
    
    def generate_explanation(
        self,
        image_features: torch.Tensor,
        detection_score: float,
        prompt: Optional[str] = None
    ) -> str:
        """
        基于视觉令牌生成自然语言解释

        Args:
            image_features: 图像视觉特征/令牌
            detection_score: 检测置信度分数 (0-1)
            prompt: 可选的文本提示

        Returns:
            str: 生成的自然语言解释文本
        """
        if self.tokenizer is None or self.llm_model is None:
            return self._fallback_explanation(detection_score)
        
        if prompt is None:
            if detection_score > 0.5:
                prompt = "Analyze this image and explain why it might be AI-generated."
            else:
                prompt = "Analyze this image and explain why it seems to be real."
        
        explanation = self.generate(prompt, image_features)
        return explanation
    
    def _fallback_explanation(self, detection_score: float) -> str:
        """
        当 LLM 不可用时的回退解释
        
        Args:
            detection_score: 检测置信度
        
        Returns:
            str: 回退解释
        """
        if detection_score > 0.5:
            base_explanation = "This image appears to be AI-generated. "
            if detection_score > 0.8:
                base_explanation += "The detection confidence is very high. "
            base_explanation += "Potential artifacts include inconsistent textures, unnatural edges, or unusual patterns."
        else:
            base_explanation = "This image appears to be real. "
            if detection_score < 0.2:
                base_explanation += "The detection confidence is very high. "
            base_explanation += "No obvious AI-generated artifacts were detected."
        
        return base_explanation
    
    def _postprocess_explanation(self, text: str) -> str:
        """
        后处理生成的解释文本，提高可读性

        Args:
            text: 原始文本

        Returns:
            str: 后处理后的文本
        """
        text = text.strip()
        
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        text = text[0].upper() + text[1:] if text else text
        
        return text
    
    def _init_lora(self) -> None:
        """
        初始化 LoRA 配置

        设置 LoraConfig，包括 rank、alpha、target_modules 等参数
        """
        if self.llm_model is not None:
            self.lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
    
    def _apply_lora(self) -> None:
        """
        应用 LoRA 到 LLM 模型

        使用 peft.get_peft_model 将 LoRA 适配器注入到 LLM 中
        """
        if self.llm_model is not None and self.lora_config is not None:
            try:
                self.llm_model = get_peft_model(self.llm_model, self.lora_config)
            except Exception as e:
                pass
