import os
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
from models.plaa_mllm import PLAAMLLM
from data.dataset_loader import AIGIDataset, val_AIGIDataset
from models.trainer import PLAAMLLMTrainer
from models.validator import PLAAMLLMValidator
from utils.metrics_utils import MetricsCalculator
from utils.log_utils import Logger
from torchvision import transforms
from peft import LoraConfig, get_peft_model


def parse_args() -> Namespace:
    """
    解析命令行参数

    Returns:
        Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="PLAA-MLLM AI Generated Image Detection")
    
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
        "--train_stage", 
        type=int, 
        default=1, 
        choices=[1, 2, 3],
        help="训练阶段: 1/2/3"
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
        default=3, 
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

    logger = Logger(name="PLAA_MLLM_Main", log_dir=path_config.logs_dir)

    # 1. 实例化基础模型
    model = PLAAMLLM(model_config, device_config, path_config)


    # 单卡训练
    # device = device_config.get_device()
    # model = model.to(device)

    # 多卡训练
    device = device_config.get_device()
    
    # 手动将非大语言模型的模块移动到主设备 cuda:0
    model.dual_stream_encoder = model.dual_stream_encoder.to(device)
    model.forensic_cross_attention = model.forensic_cross_attention.to(device)
    model.vision_token_proj = model.vision_token_proj.to(device)
    model.detection_head = model.detection_head.to(device)
    model.mask_head = model.mask_head.to(device)
    # model.llm_infer 内部的 llm_model 已经通过 device_map="auto" 分布在两张卡上了，不要动它

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        
        # 先读取 checkpoint 以判断是否包含 LoRA 阶段的权重
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        lora_weights = {k: v for k, v in state_dict.items() if 'lora' in k}
        
        # ==================== 【修复核心】按需注入 LoRA ====================
        if len(lora_weights) > 0 and args.mode in ['val', 'inference']:
            logger.info("Detected LoRA weights in checkpoint. Applying LoRA architecture...")
            lora_config = LoraConfig(
                r=model_config.lora_rank,
                lora_alpha=model_config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model.llm_infer.llm_model = get_peft_model(model.llm_infer.llm_model, lora_config)
        # =================================================================

        model.load_checkpoint(args.checkpoint)

        # ==================== 终极诊断与强制加载代码 ====================
        print("\n" + "="*60)
        print("🔍 开始强制诊断与提取 LoRA 权重...")
        
        if len(lora_weights) > 0:
            #强行将这部分参数灌入当前模型
            res = model.load_state_dict(lora_weights, strict=False)
            missing_lora = [k for k in res.missing_keys if 'lora' in k]
            print(f"2. 强制挂载执行完毕。未能成功对齐的 LoRA 参数数量: {len(missing_lora)}")
            
            # 核心验伤：检查 LoRA B 矩阵
            # 原理：LoRA 训练前，B矩阵默认全为0。如果这里求和为0，说明你第二阶段训练根本没更新参数！
            lora_b_sum = sum(p.abs().sum().item() for n, p in model.named_parameters() if 'lora_B' in n)
            print(f"3. 当前模型中 LoRA_B 的权重绝对值之和为: {lora_b_sum}")
            
            if lora_b_sum == 0.0:
                print("🚨 致命异常：LoRA_B 权重全为 0！说明你第二阶段训练时代码有 Bug，梯度根本没传给 LoRA！")
            else:
                print("✅ LoRA 权重已成功激活并生效！")
        else:
            print("🚨 致命异常：权重文件中根本不存在任何 LoRA 权重！(可能是保存时没用 peft 的机制)")
        print("="*60 + "\n")
        # ================================================================

    if args.mode == "train":
        train(model, args, logger, model_config, device_config, path_config)
    elif args.mode == "val":
        validate(model, args, logger, model_config, device_config, path_config)
    elif args.mode == "inference":
        inference(model, args, logger, model_config, device_config, path_config)


def train(
    model: PLAAMLLM, 
    args: Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    """
    训练模式

    Args:
        model: PLAA-MLLM 模型
        args: 命令行参数
        logger: 日志记录器
        model_config: 模型配置
        device_config: 设备配置
        path_config: 路径配置
    """
    logger.info(f"Starting training stage {args.train_stage}")      #日志第二条信息：开始训练阶段X

    train_dataset = AIGIDataset(
        path_config, model_config, stage=args.train_stage, split="train")   #日志第三条信息：加载训练数据集
    val_dataset = AIGIDataset(
        path_config, model_config, stage=args.train_stage, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = PLAAMLLMTrainer(model, model_config, device_config, path_config, stage=args.train_stage)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint
    )    #日志第四条信息：开始训练


def validate(
    model: PLAAMLLM, 
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

    validator = PLAAMLLMValidator(model, model_config, device_config, path_config)
    results = validator.validate(val_loader, save_results=True)

    metrics_calculator = MetricsCalculator(path_config)
    true_labels = results.get('true_labels', [])
    pred_scores = results.get('pred_scores', [])
    
    if true_labels and pred_scores:
        metrics = metrics_calculator.calculate_all_metrics(true_labels, pred_scores)
        logger.info(f"Validation Metrics: {metrics}")
        metrics_calculator.visualize_metrics(metrics, true_labels, pred_scores)


def inference(
    model: PLAAMLLM, 
    args: Namespace, 
    logger: Logger,
    model_config: ModelConfig,
    device_config: DeviceConfig,
    path_config: PathConfig
) -> None:
    """
    推理模式

    Args:
        model: PLAA-MLLM 模型
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
    model: PLAAMLLM,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    logger: Logger
) -> None:
    """
    推理单张图像

    Args:
        model: PLAA-MLLM 模型
        image_path: 图像路径
        transform: 图像变换
        device: 设备
        logger: 日志记录器
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            detection_score, explanation = model.detect_image(image_tensor)
        
        logger.info("=" * 60)
        logger.info(f"Image: {os.path.basename(image_path)}")
        logger.info(f"Detection Score: {detection_score:.4f}")
        logger.info(f"Classification: {'AI-Generated' if detection_score > 0.5 else 'Real'}")
        logger.info(f"Explanation: {explanation}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")


def _infer_batch_images(
    model: PLAAMLLM,
    image_dir: str,
    transform: transforms.Compose,
    device: torch.device,
    logger: Logger,
    path_config: PathConfig
) -> None:
    """
    批量推理图像

    Args:
        model: PLAA-MLLM 模型
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
                    detection_score, explanation = model.detect_image(image_tensor)
                
                result = {
                    'filename': filename,
                    'detection_score': float(detection_score),
                    'is_ai_generated': bool(detection_score > 0.5),
                    'explanation': explanation
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
