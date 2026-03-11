
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import CLIPVisionModel
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig


class SemanticStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.clip_model = CLIPVisionModel.from_pretrained(config.clip_model_name)
        self.intermediate_layers = config.clip_intermediate_layers

    def forward(self, x):
        """
        语义流前向传播

        Args:
            x: 输入图像张量，形状 [B, C, H, W]，其中 B=batch_size, C=channels=3, H=height, W=width

        Returns:
            Dict[str, torch.Tensor]: 包含中间层特征的字典
                - 'layer_8': 第8层特征，原始序列格式 [B, L, D]
                    (L=257 for ViT-L/14: 256 patches + 1 CLS token, D=1024)
                - 'layer_16': 第16层特征，原始序列格式 [B, L, D]
                - 'layer_24': 第24层特征，原始序列格式 [B, L, D]
            注意：如需空间特征图，需将 [B, L, D] 去除 CLS token 后 reshape 为 [B, D, H/14, W/14]
        """
        features = {}
        
        outputs = self.clip_model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        for layer_idx in self.intermediate_layers:
            if layer_idx < len(hidden_states):
                features[f'layer_{layer_idx}'] = hidden_states[layer_idx]
        
        return features





class SRMFilter(nn.Module):
    """
    空间丰富模型（SRM）高通滤波器组
    用于提取图像中的噪声残差
    """
    def __init__(self):
        super().__init__()
        # 定义 SRM 滤波器
        filter1 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32)
        
        filter2 = torch.tensor([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=torch.float32)
        
        filter3 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32)
        
        # 归一化滤波器
        filter1 /= 4.0
        filter2 /= 12.0
        filter3 /= 2.0
        
        # 扩展为 3 通道
        filter1 = filter1.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        filter2 = filter2.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        filter3 = filter3.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        
        # 创建卷积层，使用固定权重
        self.filter1 = nn.Conv2d(3, 1, kernel_size=5, padding=2, bias=False)
        self.filter2 = nn.Conv2d(3, 1, kernel_size=5, padding=2, bias=False)
        self.filter3 = nn.Conv2d(3, 1, kernel_size=5, padding=2, bias=False)
        
        # 设置权重并冻结
        self.filter1.weight.data = filter1
        self.filter2.weight.data = filter2
        self.filter3.weight.data = filter3
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 应用 SRM 滤波器
        out1 = self.filter1(x)
        out2 = self.filter2(x)
        out3 = self.filter3(x)
        
        # 拼接三个滤波器的输出
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class ArtifactStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 固定的 SRM 滤波器组
        self.srm_filter = SRMFilter()
        
        # 极浅层轻量级 CNN
        self.shallow_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        """
        底层伪影流前向传播

        Args:
            x: 输入图像张量，形状 [B, C, H, W]

        Returns:
            torch.Tensor: 伪影特征，形状 [B, 128]
        """
        # 应用 SRM 滤波器提取噪声残差
        noise_residual = self.srm_filter(x)
        
        # 极浅层 CNN 提取高频统计特征
        features = self.shallow_cnn(noise_residual)
        
        # 全局平均池化
        features = self.global_avg_pool(features)
        features = features.flatten(1)
        
        return features


class DualStreamEncoder(nn.Module):
    def __init__(self, config, device_config):
        super().__init__()
        self.config = config
        self.device_config = device_config
        self.semantic_stream = SemanticStream(config)
        self.artifact_stream = ArtifactStream(config)

    def forward(self, x):
        """
        双流编码器前向传播

        Args:
            x: 输入图像张量，形状 [B, C, H, W]

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]:
                - 语义流特征字典
                - 伪影流特征
        """
        semantic_features = self.semantic_stream(x)
        artifact_features = self.artifact_stream(x)
        return semantic_features, artifact_features

    def extract_multiscale_features(self, x):
        """
        提取多尺度融合特征

        Args:
            x: 输入图像张量，形状 [B, C, H, W]

        Returns:
            Dict[str, torch.Tensor]: 融合后的多尺度特征
        """
        semantic_features, artifact_features = self.forward(x)
        
        result = {}
        result.update(semantic_features)
        result['artifact_features'] = artifact_features
        
        return result
