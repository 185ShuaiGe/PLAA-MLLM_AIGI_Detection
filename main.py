import os
import json
import argparse
from datetime import datetime
from typing import Optional

# ==============================================================================
# 核心机制：在导入 PyTorch 之前，必须先解析参数并配置 GPU 硬件隔离环境
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """ 解析命令行参数 """
    parser = argparse.ArgumentParser(description="DS-MoME AI Generated Image Detection")
    
    # 运行模式与路径
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "val", "inference"], help="运行模式")
    parser.add_argument("--image_path", type=str, default=None, help="推理模式: 输入单张图像路径")
    parser.add_argument("--image_dir", type=str, default=None, help="推理模式: 批量图像目录")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型权重(Checkpoint)路径")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    
    # 硬件与消融实验
    parser.add_argument("--gpu_id", type=int, default=0, help="指定物理 GPU 编号 (如 0 或 1)")
    parser.add_argument("--ablation", type=str, default="final", help="指定消融组别 (A, B, C1, C2, C3, D, final)")
    
    return parser.parse_args()

# 1. 立即解析参数
args = parse_args()

# 2. 立即设置环境变量，在 PyTorch 初始化前对系统显卡进行硬隔离
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


# ==============================================================================
# 硬件隔离完成，安全导入深度学习框架及本地模块
# ==============================================================================
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.ablation_config import AblationConfig
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig

from models.ds_mome import DSMoME
from models.trainer import DSMoMETrainer
from models.validator import DSMoMEValidator
from data.dataset_loader import get_holmes_dataloaders, val_AIGIDataset
from utils.log_utils import Logger


# ==============================================================================
# 主业务流程
# ==============================================================================
def main() -> None:
    """ 项目主入口 """
    
    # --- 1. 应用消融实验配置 ---
    AblationConfig.EXPERIMENT_ID = args.ablation
    AblationConfig.apply_config()
    
    # --- 2. 初始化核心配置 ---
    model_config = ModelConfig()
    device_config = DeviceConfig()
    path_config = PathConfig()
    
    # 强制逻辑显卡编号为 0 (因外层已做物理隔离)
    device_config.gpu_ids = [0]
    device_config.cuda_visible_devices = str(args.gpu_id)

    # --- 3. 动态日志目录与 Logger 初始化 ---
    if args.mode == 'inference':
        inf_type = 'single' if args.image_path else 'batch'
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"inference_{inf_type}_{current_time}"
        
        custom_out_dir = os.path.join(path_config.outputs_dir, folder_name)
        os.makedirs(custom_out_dir, exist_ok=True)
        path_config.outputs_dir = custom_out_dir

    logger = Logger(
        name="DS_MoME_Main", 
        base_log_dir=path_config.logs_dir,
        mode=args.mode,
        exp_id=AblationConfig.EXPERIMENT_ID,
        checkpoint_path=args.checkpoint
    )

    # --- 4. 打印环境与配置信息 ---
    _print_config_info(args, logger)

    # --- 5. 显存监控 (加载前) ---
    mem_before = _get_allocated_memory(reset_peak=True)

    # --- 6. 模型初始化与设备挂载 ---
    logger.info("Initializing DS-MoME model...")
    model = DSMoME(model_config, device_config, path_config)
    device = device_config.get_device()
    
    model.dual_stream_encoder = model.dual_stream_encoder.to(device)
    model.mome_fusion = model.mome_fusion.to(device)
    model.vision_token_proj = model.vision_token_proj.to(device)
    model.detection_head = model.detection_head.to(device)

    # 加载权重
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

    # --- 7. 显存监控 (加载后) ---
    mem_after = _get_allocated_memory()

    # --- 8. 核心路由执行 ---
    if args.mode == "train":
        train(model, args, logger, model_config, device_config, path_config)
    elif args.mode == "val":
        validate(model, args, logger, model_config, device_config, path_config)
    elif args.mode == "inference":
        inference(model, args, logger, model_config, device_config, path_config)

    # --- 9. 显存总结报告 ---
    _print_memory_summary(mem_before, mem_after)


# ==============================================================================
# 核心功能模块
# ==============================================================================
def train(
    model: DSMoME,
    args: argparse.Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    logger.info("Starting training pipeline...")
    train_loader, val_loader = get_holmes_dataloaders(
        path_config, model_config, batch_size=args.batch_size
    )

    trainer = DSMoMETrainer(model, model_config, device_config, path_config)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint
    )


def validate(
    model: DSMoME, 
    args: argparse.Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    logger.info("Starting validation pipeline...")
    transform = _get_standard_transform()

    val_dataset = val_AIGIDataset(path_config.TEST_DATA_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    validator = DSMoMEValidator(model, model_config, device_config, path_config)
    validator.validate(val_loader, save_results=True)


def inference(
    model: DSMoME, 
    args: argparse.Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    logger.info("Starting inference pipeline...")
    transform = _get_standard_transform()
    device = device_config.get_device()
    model.eval()

    if args.image_path and os.path.exists(args.image_path):
        logger.info(f"Processing single image: {args.image_path}")
        _infer_single_image(model, args.image_path, transform, device, logger)
    elif args.image_dir and os.path.isdir(args.image_dir):
        logger.info(f"Processing batch images from: {args.image_dir}")
        _infer_batch_images(model, args.image_dir, transform, device, logger, path_config)
    else:
        logger.error("Please provide a valid --image_path or --image_dir for inference.")


# ==============================================================================
# 辅助工具函数
# ==============================================================================
def _get_standard_transform() -> transforms.Compose:
    """ 获取统一的图像预处理流水线 """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


def _infer_single_image(
    model: DSMoME,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    logger: Logger
) -> None:
    """ 单张图像推理具体逻辑 """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            detection_score = model.detect_image(image_tensor)
        
        logger.info("-" * 60)
        logger.info(f"Image: {os.path.basename(image_path)}")
        logger.info(f"Detection Score: {detection_score:.4f}")
        logger.info(f"Classification: {'AI-Generated' if detection_score > 0.5 else 'Real'}")
        logger.info("-" * 60)
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")


def _infer_batch_images(
    model: DSMoME,
    image_dir: str,
    transform: transforms.Compose,
    device: torch.device,
    logger: Logger,
    path_config: PathConfig
) -> None:
    """ 批量图像推理具体逻辑 """
    results = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1].lower() not in valid_extensions:
            continue
            
        image_path = os.path.join(image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                detection_score = model.detect_image(image_tensor)
            
            is_fake = bool(detection_score > 0.5)
            results.append({
                'filename': filename,
                'detection_score': float(detection_score),
                'is_ai_generated': is_fake
            })
            
            cls_text = 'AI-Generated' if is_fake else 'Real'
            logger.info(f"[{cls_text}] {filename} (Score: {detection_score:.4f})")
            
        except Exception as e:
            logger.warning(f"Failed to process {filename}: {e}")
    
    output_file = os.path.join(path_config.outputs_dir, "inference_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Batch inference results successfully saved to {output_file}")


def _print_config_info(args: argparse.Namespace, logger: Logger) -> None:
    """ 打印格式化的运行配置参数 """
    msg = "\n" + "=" * 60 + "\n🚀 [Run Configuration] 🚀\n"
    msg += f"  Experiment Group: {AblationConfig.EXPERIMENT_ID}\n"
    for arg, value in vars(args).items():
        if value is not None and arg != 'ablation':
            msg += f"  {arg}: {value}\n"
    msg += "=" * 60
    print(msg)


def _get_allocated_memory(reset_peak: bool = False) -> float:
    """ 获取当前 GPU 已分配的显存 (单位: GB) """
    if not torch.cuda.is_available():
        return 0.0
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated() / (1024 ** 3)


def _print_memory_summary(mem_before: float, mem_after: float) -> None:
    """ 打印程序运行前后的显存使用报告 """
    if not torch.cuda.is_available():
        return
        
    mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    msg = (
        "\n" + "=" * 60 + "\n"
        "📊 [Memory Tracking Report] 📊\n"
        f"1. Pre-load baseline: {mem_before:.2f} GB\n"
        f"2. Post-load memory:  {mem_after:.2f} GB\n"
        f"   (Model footprint:  {(mem_after - mem_before):.2f} GB)\n"
        f"3. Peak utilization:  {mem_peak:.2f} GB\n"
        f"   (Runtime overhead: {(mem_peak - mem_after):.2f} GB)\n"
        + "=" * 60
    )
    print(msg)


if __name__ == "__main__":
    main()