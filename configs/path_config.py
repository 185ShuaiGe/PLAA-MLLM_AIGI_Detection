
import os

class PathConfig:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    weights_dir = os.path.join(project_root, "weights")
    outputs_dir = os.path.join(project_root, "outputs")
    logs_dir = os.path.join(project_root, "logs")

    # 替换为本地绝对路径，注意使用 r"" 防止转义
    pretrained_clip_path = "D:\\cache\\huggingface_cache\\hub\\models--openai--clip-vit-large-patch14"
    pretrained_resnet_path = "D:\\cache\\torch_cache\\resnet50-0676ba61.pth"
    llm_model_name = "D:\\cache\\huggingface_cache\\hub\\models--mistralai--Mistral-7B-Instruct-v0.2\\snapshots\\63a8b081895390a26e140280378bc85ec8bce07a"    

    checkpoint_path = os.path.join(weights_dir, "plaa_mllm_checkpoint.pt")
