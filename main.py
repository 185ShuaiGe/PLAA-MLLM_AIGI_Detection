import os
from datetime import datetime
#PyTorch 在训练时容易产生显存碎片，开启 expandable_segments 可以缓解此问题，避免因为找不到连续大块显存而报错。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import argparse
import torch
# 提前唤醒多卡的 CUDA 上下文，消除 cuBLAS 警告
if torch.cuda.is_available():
    torch.tensor([0.0], device='cuda:0')
    torch.tensor([0.0], device='cuda:1')
from typing import Optional
from argparse import Namespace
from PIL import Image
from torch.utils.data import DataLoader
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.ds_mome import DSMoME
from data.dataset_loader import AIGIDataset, val_AIGIDataset
from models.trainer import DSMoMETrainer
from models.validator import PLAAMLLMValidator
from utils.metrics_utils import MetricsCalculator
from utils.log_utils import Logger
from torchvision import transforms


def parse_args() -> Namespace:
    """
    解析命令行参数

    Returns:
        Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="DS-MoME AI Generated Image Detection")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="inference", 
        choices=["train", "val", "inference"],
        help="运行模式: train/val/inference"
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        default=None, 
        help="推理模式: 输入图像路径"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default=None, 
        help="推理模式: 批量图像目录"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="模型检查点路径"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="批次大小"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=10, 
        help="训练轮数"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-5, 
        help="学习率"
    )
    
    return parser.parse_args()


def main() -> None:
    """
    主函数：项目入口
    """
    args = parse_args()

    model_config = ModelConfig()
    device_config = DeviceConfig()
    path_config = PathConfig()

    if args.mode == 'inference':
        # 判断是单张图推理还是批量推理
        inf_type = 'single' if args.image_path else 'batch'
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"inference_{inf_type}_{current_time}"
        
        # 将路径指向 outputs 目录下的新文件夹
        custom_out_dir = os.path.join(path_config.outputs_dir, folder_name)
        os.makedirs(custom_out_dir, exist_ok=True)
        
        # 指定 Logger 使用这个新文件夹
        log_dir_to_use = custom_out_dir
        
        # 顺便把 outputs_dir 覆写，这样后续生成的 inference_results.json 也会存到这里
        path_config.outputs_dir = custom_out_dir
    else:
        # 训练和验证模式保持使用默认的 logs 目录
        log_dir_to_use = path_config.logs_dir

    # 使用动态决定的路径初始化主 Logger
    logger = Logger(name="DS_MoME_Main", log_dir=log_dir_to_use)

    # 1. 实例化基础模型
    model = DSMoME(model_config, device_config, path_config)


    # 单卡训练
    # device = device_config.get_device()
    # model = model.to(device)

    # 多卡训练
    device = device_config.get_device()
    
    # 手动将非大语言模型的模块移动到主设备 cuda:0
    model.dual_stream_encoder = model.dual_stream_encoder.to(device)
    model.mome_fusion = model.mome_fusion.to(device)
    model.vision_token_proj = model.vision_token_proj.to(device)
    model.detection_head = model.detection_head.to(device)
    # model.llm_infer 内部的 llm_model 已经通过 device_map="auto" 分布在两张卡上了，不要动它

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

    if args.mode == "train":
        train(model, args, logger, model_config, device_config, path_config)
    elif args.mode == "val":
        validate(model, args, logger, model_config, device_config, path_config)
    elif args.mode == "inference":
        inference(model, args, logger, model_config, device_config, path_config)


def train(
    model: PLAAMLLM
    args: Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    """
    训练模式

    Args:
        model: DS-MoME 模型
        args: 命令行参数
        logger: 日志记录器
        model_config: 模型配置
        device_config: 设备配置
        path_config: 路径配置
    """
    logger.info("Starting training")      #日志第二条信息：开始训练

    train_dataset = AIGIDataset(
        path_config, model_config, split="train")   #日志第三条信息：加载训练数据集
    val_dataset = AIGIDataset(
        path_config, model_config, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = DSMoMETrainer(model, model_config, device_config, path_config)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint
    )    #日志第四条信息：开始训练


def validate(
    model: PLAAALLMLM, 
    args: Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    """
    验证模式

    Args:
        model: PLAA-MLLM 模型
        args: 命令行参数
        logger: 日志记录器
        model_config: 模型配置
        device_config: 设备配置
        path_config: 路径配置
    """
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

    #已经在validator中计算了指标，无需重复计算
    # metrics_calculator = MetricsCalculator(path_config)
    # true_labels = results.get('true_labels', [])
    # pred_scores = results.get('pred_scores', [])
    
    # if true_labels and pred_scores:
    #     metrics = metrics_calculator.calculate_all_metrics(true_labels, pred_scores)
    #     logger.info(f"Validation Metrics: {metrics}")
    #     metrics_calculator.visualize_metrics(metrics, true_labels, pred_scores)


def inference(
    model: PLAAALLMLM, 
    args: Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    """
    推理模式

    Args:
        model: DS-MoME 模型
        args: 命令行参数
        logger: 日志记录器
        model_config: 模型配置
        device_config: 设备配置
        path_config: 路径配置
    """
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
    """
    推理单张图像

    Args:
        model: DS-MoME 模型
        image_path: 图像路径
        transform: 图像变换
        device: 设备
        logger: 日志记录器
    """
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
    """
    批量推理图像

    Args:
        model: DS-MoME 模型
        image_dir: 图像目录
        transform: 图像变换
        device: 设备
        logger: 日志记录器
        path_config: 路径配置
    """
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
