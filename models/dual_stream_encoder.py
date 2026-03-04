
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
    
    def forward(self, inputs):
        laterals = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]
        
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest')
        
        outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        return outs


class ArtifactStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        
        self.fpn = FPN([256, 512, 1024], 256)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        底层伪影流前向传播

        Args:
            x: 输入图像张量，形状 [B, C, H, W]

        Returns:
            List[torch.Tensor]: FPN多尺度特征列表，从高分辨率到低分辨率
                - [0]: 高分辨率浅层特征，形状 [B, F, H/4, W/4]
                - [1]: 中分辨率中层特征，形状 [B, F, H/8, W/8]
                - [2]: 低分辨率深层特征，形状 [B, F, H/16, W/16]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        
        fpn_features = self.fpn([c2, c3, c4])
        
        return fpn_features


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
            Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
                - 语义流特征字典
                - 伪影流特征列表
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
