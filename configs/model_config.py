
class ModelConfig:
    clip_model_name = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/DS-MoME/cache/models--openai--clip-vit-large-patch14"  
    llm_model_name = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/DS-MoME/cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct"

    clip_dim = 1024             # CLIP 模型输出维度
    clip_intermediate_layers = [8, 16, 24]  # 选择 CLIP 的哪些层作为语义特征输入
    num_latent_queries = 128                # 生成的潜在查询数量
    latent_dim = 512                
    
    llm_dim = 4096                  # LLM 模型输出维度
    grad_accum_steps = 8            # 梯度累积步数
    max_seq_len = 1024              # 最大序列长度
