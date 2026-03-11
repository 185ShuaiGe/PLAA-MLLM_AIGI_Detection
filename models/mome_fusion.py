import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig


class MoMEFusion(nn.Module):
    """
    视觉驱动的动态专家混合 (Visual-driven MoME) 融合网络
    """
    def __init__(self, config, device_config):
        super().__init__()
        self.config = config
        self.device_config = device_config
        
        self.num_latent_queries = config.num_latent_queries
        self.latent_dim = config.latent_dim
        
        # 语义特征投影 - 考虑多尺度特征拼接
        # 假设 semantic_features 包含多个层的特征
        # 这里使用 3 层作为默认值，实际使用时会根据输入自动适应
        self.semantic_proj = nn.Linear(config.clip_dim * 3, self.latent_dim)
        
        # 伪影特征投影
        self.artifact_proj = nn.Linear(128, self.latent_dim)
        
        # 构建专家池
        self.experts = nn.ModuleDict({
            'semantic_geometry': nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            ),
            'low_level_texture': nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            ),
            'lighting_consistency': nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
        })
        
        # 动态软路由门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, len(self.experts)),
            nn.Softmax(dim=-1)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(self.latent_dim, self.latent_dim)
        
        # 初始化 latent queries
        self.latent_queries = nn.Parameter(
            torch.randn(self.num_latent_queries, self.latent_dim)
        )
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
    
    def forward(self, semantic_features, artifact_features):
        """
        MoME 融合前向传播

        Args:
            semantic_features: 语义流特征字典
            artifact_features: 伪影流特征

        Returns:
            torch.Tensor: 融合后的视觉取证 tokens
        """
        first_feat = list(semantic_features.values())[0]
        device = first_feat.device
        batch_size = first_feat.size(0)
        
        # 处理语义特征 - 融合多尺度特征
        semantic_feats = []
        for layer_name, feat in semantic_features.items():
            cls_token = feat[:, 0, :]  # 提取每一层的 CLS token
            semantic_feats.append(cls_token)
        
        # 拼接多尺度特征
        semantic_feat = torch.cat(semantic_feats, dim=1)
        semantic_feat = self.semantic_proj(semantic_feat)
        
        # 处理伪影特征
        artifact_feat = self.artifact_proj(artifact_features)
        
        # 特征拼接形成联合视觉序列
        joint_feat = torch.cat([semantic_feat, artifact_feat], dim=1)
        
        # 动态软路由门控
        gate_weights = self.gate_network(joint_feat)
        
        # 专家混合
        expert_outputs = []
        for expert_name, expert in self.experts.items():
            expert_out = expert(joint_feat)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, latent_dim]
        gate_weights = gate_weights.unsqueeze(-1)  # [B, num_experts, 1]
        
        # 加权融合
        fused_feat = torch.sum(expert_outputs * gate_weights, dim=1)  # [B, latent_dim]
        
        # 应用输出投影
        fused_feat = self.output_proj(fused_feat)
        
        # 扩展为 tokens 格式
        latent_queries = self.latent_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        fused_tokens = latent_queries + fused_feat.unsqueeze(1)
        
        return fused_tokens
