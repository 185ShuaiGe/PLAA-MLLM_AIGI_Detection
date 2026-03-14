import os
import argparse
from datetime import datetime

# =========================================================================
# 核心修复：必须在 import torch 和加载任何深度学习库之前解析参数并设置显卡可见性
# =========================================================================
def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="DS-MoME AI Generated Image Detection")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="inference", 
        choices=["train", "val", "inference"],
        help="运行模式: train/val/inference"
    )
    parser.add_argument("--image_path", type=str, default=None, help="推理模式: 输入图像路径")
    parser.add_argument("--image_dir", type=str, default=None, help="推理模式: 批量图像目录")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的 GPU 编号 (例如 0 或 1)')
    
    return parser.parse_args()

# 1. 立即解析参数
args = parse_args()

# 2. 立即设置环境变量，在 PyTorch 初始化前对系统显卡进行硬隔离
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# =========================================================================
# 环境变量隔离完成后，现在才可以安全地导入 PyTorch 和模型库
# =========================================================================
import torch
from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.ds_mome import DSMoME
from data.dataset_loader import get_holmes_dataloaders, val_AIGIDataset
from models.trainer import DSMoMETrainer
from models.validator import DSMoMEValidator
from utils.log_utils import Logger
from torchvision import transforms


def main() -> None:
    """
    主函数：项目入口
    """
    # 此时 args 已经作为全局变量存在，直接使用即可
    model_config = ModelConfig()
    device_config = DeviceConfig()
    
    # 既然已经用环境变量隔离，PyTorch内部只能看到1张卡，所以其逻辑序号必须设置为 0
    device_config.gpu_ids = [0]
    device_config.cuda_visible_devices = str(args.gpu_id)
    
    path_config = PathConfig()

    if args.mode == 'inference':
        inf_type = 'single' if args.image_path else 'batch'
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"inference_{inf_type}_{current_time}"
        
        custom_out_dir = os.path.join(path_config.outputs_dir, folder_name)
        os.makedirs(custom_out_dir, exist_ok=True)
        log_dir_to_use = custom_out_dir
        path_config.outputs_dir = custom_out_dir
    else:
        log_dir_to_use = path_config.logs_dir

    logger = Logger(name="DS_MoME_Main", log_dir=log_dir_to_use)


    # ================== 👇 打印命令行输入参数 👇 ==================
    config_msg = "=" * 60 + "\nRun Configuration:\n"
    for arg, value in vars(args).items():
        if value is not None: 
            config_msg += f"  {arg}: {value}\n"
    config_msg += "=" * 60
    print(config_msg)
    # ==============================================================

    # 🌟 显存记录点 1：加载大模型前
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() # 删除了 0，默认使用当前设备
        mem_before = torch.cuda.memory_allocated() / 1024**3
    else:
        mem_before = 0.0

    model = DSMoME(model_config, device_config, path_config)

    device = device_config.get_device()
    
    model.dual_stream_encoder = model.dual_stream_encoder.to(device)
    model.mome_fusion = model.mome_fusion.to(device)
    model.vision_token_proj = model.vision_token_proj.to(device)
    model.detection_head = model.detection_head.to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

    # 🌟 显存记录点 2：加载大模型后
    mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    # 开始执行训练或推理
    if args.mode == "train":
        train(model, args, logger, model_config, device_config, path_config)
    elif args.mode == "val":
        validate(model, args, logger, model_config, device_config, path_config)
    elif args.mode == "inference":
        inference(model, args, logger, model_config, device_config, path_config)

    # 🌟 显存记录点 3：运行结束后获取全局峰值显存（即训练时的最大占用）
    if torch.cuda.is_available():
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        summary_msg = (
            "\n" + "=" * 60 + "\n"
            "🚀 [显存占用追踪报告] 🚀\n"
            f"1. 加载模型前基础占用: {mem_before:.2f} GB\n"
            f"2. 加载全模型后占用:   {mem_after:.2f} GB\n"
            f"   (模型净重: {(mem_after - mem_before):.2f} GB)\n"
            f"3. 训练时峰值占用:     {mem_peak:.2f} GB\n"
            f"   (训练计算额外开销: {(mem_peak - mem_after):.2f} GB)\n"
            + "=" * 60
        )
        print(summary_msg)

def train(
    model: DSMoME,
    args: argparse.Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    logger.info("Starting training")
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
    logger.info("Starting validation")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    val_dataset = val_AIGIDataset(
        path_config.TEST_DATA_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    validator = DSMoMEValidator(model, model_config, device_config, path_config)
    results = validator.validate(val_loader, save_results=True)


def inference(
    model: DSMoME, 
    args: argparse.Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    logger.info("Starting inference")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    device = device_config.get_device()
    model.eval()

    if args.image_path and os.path.exists(args.image_path):
        logger.info(f"Processing single image: {args.image_path}")
        _infer_single_image(model, args.image_path, transform, device, logger)
    elif args.image_dir and os.path.isdir(args.image_dir):
        logger.info(f"Processing batch images from: {args.image_dir}")
        _infer_batch_images(model, args.image_dir, transform, device, logger, path_config)
    else:
        logger.error("Please provide either --image_path or --image_dir for inference")


def _infer_single_image(
    model: DSMoME,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    logger: Logger
) -> None:
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            detection_score = model.detect_image(image_tensor)
        
        logger.info("=" * 60)
        logger.info(f"Image: {os.path.basename(image_path)}")
        logger.info(f"Detection Score: {detection_score:.4f}")
        logger.info(f"Classification: {'AI-Generated' if detection_score > 0.5 else 'Real'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")


def _infer_batch_images(
    model: DSMoME,
    image_dir: str,
    transform: transforms.Compose,
    device: torch.device,
    logger: Logger,
    path_config: PathConfig
) -> None:
    import json
    
    results = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_path = os.path.join(image_dir, filename)
            
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    detection_score = model.detect_image(image_tensor)
                
                result = {
                    'filename': filename,
                    'detection_score': float(detection_score),
                    'is_ai_generated': bool(detection_score > 0.5)
                }
                results.append(result)
                
                logger.info(f"{filename}: Score={detection_score:.4f}, Classification={'AI-Generated' if detection_score > 0.5 else 'Real'}")
                
            except Exception as e:
                logger.warning(f"Failed to process {filename}: {e}")
    
    output_file = os.path.join(path_config.outputs_dir, "inference_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Batch inference results saved to {output_file}")


if __name__ == "__main__":
    main()