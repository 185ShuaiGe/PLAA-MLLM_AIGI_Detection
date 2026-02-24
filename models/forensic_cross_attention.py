
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig


class ForensicCrossAttention(nn.Module):
    def __init__(self, config, device_config):
        super().__init__()
        self.config = config
        self.device_config = device_config
        
        self.num_latent_queries = config.num_latent_queries
        self.latent_dim = config.latent_dim
        
        self._init_latent_queries()
        
        self.semantic_proj = nn.Linear(config.clip_dim, self.latent_dim)
        self.artifact_projs = nn.ModuleList([
            nn.Linear(dim, self.latent_dim) for dim in config.artifact_dims
        ])
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=config.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=config.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 4),
            nn.GELU(),
            nn.Linear(self.latent_dim * 4, self.latent_dim),
            nn.Dropout(0.1)
        )
        
        self.norm1 = nn.LayerNorm(self.latent_dim)
        self.norm2 = nn.LayerNorm(self.latent_dim)
        self.norm3 = nn.LayerNorm(self.latent_dim)
        
        self.text_guidance_proj = None
        if config.use_text_guidance:
            self.text_guidance_proj = nn.Linear(config.llm_dim, self.latent_dim)

    def forward(self, semantic_features, artifact_features, text_guidance=None):
        first_feat = list(semantic_features.values())[0]
        device = first_feat.device
        batch_size = first_feat.size(0)
        
        keys, values = self._align_features(semantic_features, artifact_features)
        
        queries = self.latent_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        
        if text_guidance is not None and self.text_guidance_proj is not None:
            text_proj = self.text_guidance_proj(text_guidance)
            keys = torch.cat([keys, text_proj], dim=1)
            values = torch.cat([values, text_proj], dim=1)
        
        attn_output, _ = self.cross_attention(
            query=queries,
            key=keys,
            value=values
        )
        queries = self.norm1(queries + attn_output)
        
        self_attn_output, _ = self.self_attention(
            query=queries,
            key=queries,
            value=queries
        )
        queries = self.norm2(queries + self_attn_output)
        
        ffn_output = self.ffn(queries)
        queries = self.norm3(queries + ffn_output)
        
        return queries

    def _init_latent_queries(self):
        self.latent_queries = nn.Parameter(
            torch.randn(self.num_latent_queries, self.latent_dim)
        )
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

    def _align_features(self, semantic_features, artifact_features):
        semantic_aligned = []
        for layer_name, feat in semantic_features.items():
            if feat.dim() == 3:
                proj_feat = self.semantic_proj(feat)
                semantic_aligned.append(proj_feat)
        
        artifact_aligned = []
        num_projs = len(self.artifact_projs)
        for idx, feat in enumerate(artifact_features):
            if feat.dim() == 4:
                B, C, H, W = feat.shape
                feat = feat.flatten(2).transpose(1, 2)
            if idx < num_projs:
                proj_feat = self.artifact_projs[idx](feat)
                artifact_aligned.append(proj_feat)
        
        all_features = semantic_aligned + artifact_aligned
        
        if all_features:
            keys = torch.cat(all_features, dim=1)
            values = keys
        else:
            first_feat = list(semantic_features.values())[0]
            batch_size = first_feat.size(0)
            keys = torch.zeros(batch_size, 1, self.latent_dim, device=first_feat.device)
            values = keys
        
        return keys, values

