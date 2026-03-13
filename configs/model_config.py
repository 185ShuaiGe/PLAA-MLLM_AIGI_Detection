
class ModelConfig:
    #local
    #clip_model_name = "D:\\cache\\huggingface_cache\\hub\\models--openai--clip-vit-large-patch14"  
    # schoolserver
    clip_model_name = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/PLAA-MLLM_AIGI_Detection/cache/models--openai--clip-vit-large-patch14"  

    clip_dim = 1024
    clip_intermediate_layers = [8, 16, 24]
    resnet_depth = 50
    fpn_channels = 256
    artifact_dims = [256, 256, 256]
    num_latent_queries = 128
    latent_dim = 512
    num_attention_heads = 8

    #local
    # llm_model_name = "D:\\cache\\huggingface_cache\\hub\\models--meta-llama--Meta-Llama-3.1-8B-Instruct"    
    # schoolserver
    llm_model_name = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/PLAA-MLLM_AIGI_Detection/cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
    
    llm_dim = 4096
    use_text_guidance = True
    grad_accum_steps = 8
    max_seq_len = 1024
    hidden_dim = 1024
    dropout = 0.1
    
    # Ablation study switches
    use_resnet_artifact = False
    use_cross_attention = False
    enable_text_loss = False
