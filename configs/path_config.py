
import os

class PathConfig:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    weights_dir = os.path.join(project_root, "weights")
    outputs_dir = os.path.join(project_root, "outputs")
    logs_dir = os.path.join(project_root, "logs")

    # 替换为本地绝对路径，注意使用 r"" 防止转义
    # pretrained_clip_path = "D:\\cache\\huggingface_cache\\hub\\models--openai--clip-vit-large-patch14"
    # pretrained_resnet_path = "D:\\cache\\torch_cache\\resnet50-0676ba61.pth"
    # llm_model_name = "D:\\cache\\huggingface_cache\\hub\\models--meta-llama--Meta-Llama-3.1-8B-Instruct"    

    # schoolserver
    pretrained_clip_path = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/PLAA-MLLM_AIGI_Detection/cache/models--openai--clip-vit-large-patch14"
    pretrained_resnet_path = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/PLAA-MLLM_AIGI_Detection/cache/resnet50-0676ba61.pth"
    llm_model_name = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/PLAA-MLLM_AIGI_Detection/cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct"


    checkpoint_path = os.path.join(weights_dir, "plaa_mllm_checkpoint.pt")

    # 在 configs/path_config.py 中，用于测试的路径
    TEST_DATA_DIR = "/data/Disk_A/wangxinchang/Datasets/val/progan"
    CHECKPOINT_PATH = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/PLAA-MLLM_AIGI_Detection/weights/checkpoint_stage2_best.pt"
