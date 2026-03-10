
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig


class LLMInference(nn.Module):
    def __init__(self, config: ModelConfig, device_config: DeviceConfig):
        super().__init__()
        self.config = config
        self.device_config = device_config
        # --- 修改部分：增加模型加载逻辑 ---
        try:
            print(f"Loading LLM from {config.llm_model_name}...")
            # 配置 4-bit 量化
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name,
                # torch_dtype=torch.float16,
                quantization_config=quantization_config, # 使用量化配置
                device_map="auto",        
                max_memory={0: "0GiB", 1: "23GiB"} #限制 GPU 0 的最大显存使用量，强制剩余部分溢出到 GPU 1 
            )
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.llm_model = None 
        # ------------------------------
        self.tokenizer = None
        self.lora_config = None
        
        self.device = device_config.get_device()
        self._init_tokenizer()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _init_tokenizer(self) -> None:
        """
        初始化 Tokenizer
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name, legacy=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            self.tokenizer = None
    
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
        
        # 获取 LLM 输入层当前所在的设备 (通常是 cuda:1)
        llm_device = self.llm_model.device
        
        # 将文本输入转移到 LLM 所在的设备
        input_ids = input_ids.to(llm_device)
        attention_mask = attention_mask.to(llm_device)
        
        inputs_embeds = None
        fused_attention_mask = attention_mask
        
        if vision_tokens is not None:
            if hasattr(self.llm_model, 'get_input_embeddings'):
                text_embeds = self.llm_model.get_input_embeddings()(input_ids)
                
                # 将 vision_tokens 转移到 text_embeds 所在的同一张显卡上, 同时将float32转换为float16（如果 LLM 使用了 float16），显式对齐数据类型
                vision_tokens = vision_tokens.to(device = text_embeds.device, dtype = text_embeds.dtype)
                inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
                
                vision_mask = torch.ones(
                    (vision_tokens.size(0), vision_tokens.size(1)), 
                    dtype=attention_mask.dtype, 
                    device=text_embeds.device
                )
                
                # 【关键修复】确保 attention_mask 也在同一张显卡上
                fused_attention_mask = torch.cat([vision_mask, attention_mask.to(text_embeds.device)], dim=1)
        
        # 前向传播 (Transformers 的 device_map 会自动处理后续层之间的显卡调度)
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
        if self.tokenizer is None or self.llm_model is None:
            return None # 改为返回 None，把后备处理交给外层
        
        try:
            # 获取 LLM 当前所在的设备
            llm_device = self.llm_model.device
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(llm_device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            inputs_embeds = None
            if vision_tokens is not None:
                if hasattr(self.llm_model, 'get_input_embeddings'):
                    text_embeds = self.llm_model.get_input_embeddings()(input_ids)
                    
                    # 【核心修复 1】: 将 vision_tokens 转移到 text_embeds 的设备和类型上
                    vision_tokens = vision_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)
                    inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
                    
                    # 【核心修复 2】: 确保 vision_mask 也在相同的设备上
                    vision_mask = torch.ones((vision_tokens.size(0), vision_tokens.size(1)), 
                                            dtype=attention_mask.dtype, device=text_embeds.device)
                    fused_attention_mask = torch.cat([vision_mask, attention_mask.to(text_embeds.device)], dim=1)
                    attention_mask = fused_attention_mask
            
            # 在 generate 函数中修改生成参数
            generate_kwargs = {
                'max_new_tokens': 128,         # 从256降低到128，防止解释完毕后强行凑字数
                'do_sample': False,            # 开启贪心解码，必须设为 False（关闭随机采样）
                'repetition_penalty': 1.1,     # 从1.2降为1.1，惩罚过高会导致模型造出不存在的生僻词
                'pad_token_id': self.tokenizer.eos_token_id
            }
            # 注意：如果 do_sample=False，就不要再传 temperature 和 top_p 参数了，否则会报错。
            
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
            # 打印错误，这样以后出现问题就不会再“悄悄失败”了
            print(f"\n Generation Error: {e}\n")
            return None   

    def generate_explanation(
        self,
        image_features: torch.Tensor,
        detection_score: float,
        prompt: Optional[str] = None
    ) -> str:
        
        if self.tokenizer is None or self.llm_model is None:
            return self._fallback_explanation(detection_score)
        
        if prompt is None:
            if detection_score > 0.5:
                prompt = "Analyze this image and explain why it might be AI-generated."
            else:
                prompt = "Analyze this image and explain why it seems to be real."
        
        explanation = self.generate(prompt, image_features)
        
        # 【核心修复 3】: 如果生成失败（返回了 None），我们用正确的 detection_score 进行兜底
        if explanation is None:
            return self._fallback_explanation(detection_score)
            
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
                print(f"Error applying LoRA: {e}")
