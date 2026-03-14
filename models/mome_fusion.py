import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.ablation_config import AblationConfig

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

        # 传统 MLP 融合 - 考虑消融实验
        self.traditional_mlp_fusion = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
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
        ref_feat = list(semantic_features.values())[0] if semantic_features is not None else artifact_features
        device = ref_feat.device
        batch_size = ref_feat.size(0)
        
        # 特征预处理
        semantic_feat = None
        if semantic_features is not None:
            semantic_feats = [feat[:, 0, :] for feat in semantic_features.values()]
            semantic_feat = self.semantic_proj(torch.cat(semantic_feats, dim=1))
            
        artifact_feat = None
        if artifact_features is not None:
            artifact_feat = self.artifact_proj(artifact_features)
            
        # ================= 动态消融路由 =================
        if AblationConfig.fusion_strategy == "none":
            # 基线 A 或 B (单流直接穿透)
            fused_feat = semantic_feat if semantic_feat is not None else artifact_feat
            
        elif AblationConfig.fusion_strategy == "mlp":
            # 消融 D: 传统双层 MLP 硬拼接
            joint_feat = torch.cat([semantic_feat, artifact_feat], dim=1)
            fused_feat = self.traditional_mlp_fusion(joint_feat)
            
        else:
            # 最终形态 (final) 和 消融 C: 动态 MoME 融合
            joint_feat = torch.cat([semantic_feat, artifact_feat], dim=1)
            gate_weights = self.gate_network(joint_feat)
            
            expert_outputs = [expert(joint_feat) for expert in self.experts.values()]
            expert_outputs = torch.stack(expert_outputs, dim=1) 
            gate_weights = gate_weights.unsqueeze(-1) 
            
            fused_feat = torch.sum(expert_outputs * gate_weights, dim=1) 
            fused_feat = self.output_proj(fused_feat)
            
        # 统一打包为 Token 格式喂给 LLM
        latent_queries = self.latent_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        fused_tokens = latent_queries + fused_feat.unsqueeze(1)
        
        return fused_tokens
