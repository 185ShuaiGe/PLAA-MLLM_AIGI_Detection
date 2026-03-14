import os
import argparse

# =========================================================
# 0. 解析参数并隔离显卡 (必须在 import torch 之前执行)
# =========================================================
parser = argparse.ArgumentParser(description="Test DSMoME on FDMAS dataset")
parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的 GPU 编号 (例如 0 或 1)')
args = parser.parse_args()

# 强制系统只对当前 Python 进程暴露指定的显卡，逻辑编号永远为 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================================
# 导入深度学习库 (此时 PyTorch 只能看到你指定的那张卡)
# =========================================================
import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# 导入你项目的相关配置和模型
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.ds_mome import DSMoME
from peft import LoraConfig, get_peft_model

# 忽略烦人的警告
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 配置参数
# ---------------------------------------------------------
# FDMAS 测试集根目录
TEST_ROOT = '/data/Disk_A/wangxinchang/Datasets/fdmas/test/'

# 你的训练权重路径
MODEL_PATH = '/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/DS-MoME/weights/checkpoint_best.pt'

# 批大小 (由于 MLLM 显存占用较大，建议调小，如 8 或 16)
BATCH_SIZE = 4

# 统一的文本 Prompt
TEXT_PROMPT = "<image>\nAnalyze this image and determine if it is real or AI-generated. Please provide your reasoning."

# ---------------------------------------------------------
# 2. 主程序
# ---------------------------------------------------------
def main():
    print(f"🚀 开始测试 FDMAS 数据集...")
    print(f"🖥️  使用物理 GPU 编号: {args.gpu_id}")
    print(f"📂 数据集路径: {TEST_ROOT}")
    print(f"⚖️  模型路径: {MODEL_PATH}")
    
    # 初始化配置
    model_config = ModelConfig()
    device_config = DeviceConfig()
    
    # 强制将配置中的设备对齐到当前唯一的逻辑卡 0
    device_config.gpu_ids = [0]
    device_config.cuda_visible_devices = str(args.gpu_id)
    
    path_config = PathConfig()
    # 获取设备 (此时将返回 cuda:0)
    device = device_config.get_device()

    # 1. 初始化模型
    print("⏳ 正在初始化 DSMoME 模型...")
    model = DSMoME(model_config, device_config, path_config)

    # 手动将非大语言模型的模块移动到主设备 (适配你原本 main.py 中的设定)
    model.dual_stream_encoder = model.dual_stream_encoder.to(device)
    model.mome_fusion = model.mome_fusion.to(device)
    model.vision_token_proj = model.vision_token_proj.to(device)
    model.detection_head = model.detection_head.to(device)
    # 如果你在后续更新中删除了 mask_head，请把下面这行注释掉；如果还在则保留
    if hasattr(model, 'mask_head') and model.mask_head is not None:
        model.mask_head = model.mask_head.to(device)

    # 2. 加载权重 (包含自动兼容 Stage 2 LoRA 的逻辑)
    print("⏳ 正在加载权重...")
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        lora_weights = {k: v for k, v in state_dict.items() if 'lora' in k}
        
        # 如果你后续用 stage 2 的权重测试，这部分会自动加载 LoRA
        if len(lora_weights) > 0:
            print("Detected LoRA weights. Applying LoRA architecture...")
            lora_config = LoraConfig(
                r=model_config.lora_rank,
                lora_alpha=model_config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model.llm_infer.llm_model = get_peft_model(model.llm_infer.llm_model, lora_config)

        model.load_checkpoint(MODEL_PATH)
    else:
        print(f"❌ 找不到权重文件: {MODEL_PATH}")
        return

    model.eval()
    print("✅ 模型加载完成！")

    # 3. 数据预处理 (与你项目 validation 和 inference 保持一致)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # 4. 获取所有子数据集文件夹
    sub_datasets = sorted([d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))])
    
    print("\n" + "="*70)
    # 新增 RACC、FACC 列
    print(f"{'Dataset':<25} | {'ACC (%)':<10} | {'RACC (%)':<10} | {'FACC (%)':<10} | {'AP (%)':<10}")
    print("-" * 70)

    all_acc = []
    all_ap = []
    all_racc = []  # 存储所有子数据集的 RACC
    all_facc = []  # 存储所有子数据集的 FACC

    # 5. 循环测试每个子数据集
    for dataset_name in sub_datasets:
        dataset_path = os.path.join(TEST_ROOT, dataset_name)
        
        # 使用 ImageFolder 自动识别 0_real 和 1_fake
        try:
            dataset = ImageFolder(root=dataset_path, transform=transform)
        except Exception as e:
            # 适配新增列的错误打印格式
            print(f"{dataset_name:<25} | Error      | Error      | Error      | Error: {e}")
            continue

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        y_true = []
        y_pred = []
        
        # 推理循环
        with torch.no_grad():
            # 使用 tqdm 包装 dataloader 显示进度条（可选，如果不想要可以删去 tqdm()）
            for images, labels in dataloader:
                images = images.to(device)
                
                # 构造 DSMoME 需要的文本输入
                prompts = [TEXT_PROMPT] * images.size(0)
                
                # 开启混合精度以节省显存并避免报错 (与 trainer.py 保持一致)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 前向传播提取 logits
                    outputs = model(images, prompts, text_guidance=prompts)
                    logits = outputs.get('detection_logits', None)
                    
                    if logits is not None:
                        # 计算概率 (Sigmoid) 并转为一维数组
                        probs = torch.sigmoid(logits.float()).squeeze()
                        
                        # 处理 batch_size=1 的情况
                        if probs.ndim == 0:
                            probs = probs.unsqueeze(0)
                            
                        y_pred.extend(probs.cpu().tolist())
                        y_true.extend(labels.tolist())

        if len(y_true) == 0:
            continue

        # 计算基础指标
        y_true = np.array(y_true)
        y_pred = np.array(y_pred).reshape(-1, 1)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        
        # ---------------------- 新增：计算 RACC 和 FACC ----------------------
        # RACC: Real样本（标签0）的准确率 (预测<=0.5为正确)
        real_mask = y_true == 0
        real_total = np.sum(real_mask)
        racc = (np.sum((y_pred[real_mask] <= 0.5).astype(int)) / real_total) if real_total > 0 else 0.0
        
        # FACC: Fake样本（标签1）的准确率 (预测>0.5为正确)
        fake_mask = y_true == 1
        fake_total = np.sum(fake_mask)
        facc = (np.sum((y_pred[fake_mask] > 0.5).astype(int)) / fake_total) if fake_total > 0 else 0.0
        # ---------------------------------------------------------------------

        # 收集所有指标
        all_acc.append(acc)
        all_ap.append(ap)
        all_racc.append(racc)
        all_facc.append(facc)
        
        # 打印当前子数据集结果（新增 RACC、FACC）
        print(f"{dataset_name:<25} | {acc*100:5.2f}      | {racc*100:5.2f}      | {facc*100:5.2f}      | {ap*100:5.2f}")

    # 6. 打印平均值
    print("-" * 70)
    if all_acc:
        mean_acc = np.mean(all_acc) * 100
        mean_ap = np.mean(all_ap) * 100
        mean_racc = np.mean(all_racc) * 100  # 平均 RACC
        mean_facc = np.mean(all_facc) * 100  # 平均 FACC
        print(f"{'MEAN':<25} | {mean_acc:5.2f}      | {mean_racc:5.2f}      | {mean_facc:5.2f}      | {mean_ap:5.2f}")
    print("=" * 70)

if __name__ == '__main__':
    main()