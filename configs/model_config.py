
class ModelConfig:
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_dim = 1024
    clip_intermediate_layers = [8, 16, 24]
    resnet_depth = 50
    fpn_channels = 256
    artifact_dims = [256, 512, 1024]
    num_latent_queries = 128
    latent_dim = 512
    num_attention_heads = 8
    cross_attention_heads = 8
    cross_attention_layers = 6
    llm_model_name = "meta-llama/Llama-2-7b-hf"
    llm_dim = 4096
    use_text_guidance = True
    lora_rank = 8
    lora_alpha = 32
    max_seq_len = 512
    hidden_dim = 1024
    dropout = 0.1
