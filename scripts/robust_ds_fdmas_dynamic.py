import os
import sys
import argparse
import logging
import datetime
import io
import warnings
from PIL import Image, ImageFilter

# =========================================================
# 0. 路径挂载与系统配置
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

parser = argparse.ArgumentParser(description="Dynamic Robustness Test DSMoME")
parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的 GPU 编号')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.ds_mome import DSMoME

warnings.filterwarnings('ignore')

# =========================================================
# 1. 自定义动态扰动 Transforms (核心创新点)
# =========================================================
class JPEGPerturbation:
    """在内存中模拟 JPEG 压缩降质"""
    def __init__(self, quality):
        self.quality = quality
        
    def __call__(self, img):
        # 必须转为 RGB，否则带 Alpha 通道的图无法保存为 JPEG
        img = img.convert('RGB')
        buffer = io.BytesIO()
        # 将图片以指定质量保存到内存缓存区
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        # 从缓存区重新读取，此时图像已经带有 JPEG 压缩伪影
        return Image.open(buffer)

class BlurPerturbation:
    """动态高斯模糊"""
    def __init__(self, radius):
        self.radius = radius
        
    def __call__(self, img):
        img = img.convert('RGB')
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

class IdentityTransform:
    """不做任何改变 (用于测试基线原图)"""
    def __call__(self, img):
        return img.convert('RGB')

# =========================================================
# 2. 双端日志重定向设置
# =========================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'robust_test_fdmas_dynamic.log')
sys.stdout = Logger(LOG_FILE)

# =========================================================
# 3. 基础参数与扰动配置
# =========================================================
CLEAN_TEST_ROOT = '/data/Disk_A/wangxinchang/Datasets/fdmas/test'
MODEL_PATH = os.path.join(PROJECT_ROOT, 'weights/checkpoint_best.pt')
BATCH_SIZE = 8  # 如果显存够，可以调大
NUM_WORKERS = 4 # 关键参数：通过多进程掩盖 CPU 施加扰动的耗时
TEXT_PROMPT = "<image>\nAnalyze this image and determine if it is real or AI-generated. Please provide your reasoning."

# 定义需要遍历测试的各种扰动情况
PERTURBATION_CONFIGS = {
    # 'Clean (No Perturbation)': IdentityTransform(),
    'JPEG_90': JPEGPerturbation(quality=90),
    'JPEG_75': JPEGPerturbation(quality=75),
    'JPEG_50': JPEGPerturbation(quality=50),
    'Blur_Sigma1.0': BlurPerturbation(radius=1.0),
    'Blur_Sigma2.0': BlurPerturbation(radius=2.0),
    'Blur_Sigma3.0': BlurPerturbation(radius=3.0),
}

# 基础的预处理 (模型必需的标准化)
base_transform_ops = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
]

# =========================================================
# 4. 主程序
# =========================================================
def main():
    print("\n" + "#"*70)
    print(f"🚀 [START] 动态鲁棒性测试 (0硬盘占用) - 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 干净测试集源路径: {CLEAN_TEST_ROOT}")
    print(f"🖥️  使用 GPU: {args.gpu_id} | Num Workers: {NUM_WORKERS}")
    print("#"*70 + "\n")
    
    model_config = ModelConfig()
    device_config = DeviceConfig()
    device_config.gpu_ids = [0]
    device_config.cuda_visible_devices = str(args.gpu_id)
    path_config = PathConfig()
    device = device_config.get_device()

    print("⏳ 正在初始化 DSMoME 模型并加载权重...")
    model = DSMoME(model_config, device_config, path_config)
    model.dual_stream_encoder = model.dual_stream_encoder.to(device)
    model.mome_fusion = model.mome_fusion.to(device)
    model.vision_token_proj = model.vision_token_proj.to(device)
    model.detection_head = model.detection_head.to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_checkpoint(MODEL_PATH)
    else:
        print(f"❌ 找不到权重文件: {MODEL_PATH}")
        return

    model.eval()
    print("✅ 模型就绪！\n")

    # 获取干净数据集下的所有 16 个类别子文件夹
    sub_datasets = sorted([d for d in os.listdir(CLEAN_TEST_ROOT) if os.path.isdir(os.path.join(CLEAN_TEST_ROOT, d))])

    # ---------------------------------------------------------
    # 核心测试循环：遍历每一种扰动配置
    # ---------------------------------------------------------
    for pert_name, dynamic_transform in PERTURBATION_CONFIGS.items():
        print("\n" + "★"*70)
        print(f"🧪 当前施加动态扰动: 【 {pert_name} 】")
        print("★"*70)

        # 组合当前扰动操作与基础标准化操作
        current_transform = transforms.Compose([
            dynamic_transform,  # 第一步：施加图像降质
            *base_transform_ops # 第二步：执行 Resize, ToTensor, Normalize
        ])

        print("\n" + "="*70)
        print(f"{'Dataset Category':<25} | {'ACC (%)':<10} | {'RACC (%)':<10} | {'FACC (%)':<10} | {'AP (%)':<10}")
        print("-" * 70)

        all_acc, all_ap, all_racc, all_facc = [], [], [], []

        for dataset_name in sub_datasets:
            dataset_path = os.path.join(CLEAN_TEST_ROOT, dataset_name)
            
            try:
                # 每次读取都应用包含当前扰动操作的 transform
                dataset = ImageFolder(root=dataset_path, transform=current_transform)
            except Exception as e:
                print(f"{dataset_name:<25} | Error      | Error      | Error      | Error: {e}")
                continue

            # 启用多进程加载，掩盖 CPU 处理降质的时间
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            
            y_true, y_pred = [], []
            
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(device)
                    prompts = [TEXT_PROMPT] * images.size(0)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(images, prompts, text_guidance=prompts)
                        logits = outputs.get('detection_logits', None)
                        
                        if logits is not None:
                            probs = torch.sigmoid(logits.float()).squeeze()
                            if probs.ndim == 0:
                                probs = probs.unsqueeze(0)
                                
                            y_pred.extend(probs.cpu().tolist())
                            y_true.extend(labels.tolist())

            if len(y_true) == 0:
                continue

            y_true = np.array(y_true)
            y_pred = np.array(y_pred).reshape(-1, 1)
            
            acc = accuracy_score(y_true, y_pred > 0.5)
            ap = average_precision_score(y_true, y_pred)
            
            real_mask = y_true == 0
            real_total = np.sum(real_mask)
            racc = (np.sum((y_pred[real_mask] <= 0.5).astype(int)) / real_total) if real_total > 0 else 0.0
            
            fake_mask = y_true == 1
            fake_total = np.sum(fake_mask)
            facc = (np.sum((y_pred[fake_mask] > 0.5).astype(int)) / fake_total) if fake_total > 0 else 0.0

            all_acc.append(acc)
            all_ap.append(ap)
            all_racc.append(racc)
            all_facc.append(facc)
            
            print(f"{dataset_name:<25} | {acc*100:5.2f}      | {racc*100:5.2f}      | {facc*100:5.2f}      | {ap*100:5.2f}")

        # 打印当前扰动下的 16 个类别的平均指标
        print("-" * 70)
        if all_acc:
            mean_acc, mean_ap = np.mean(all_acc) * 100, np.mean(all_ap) * 100
            mean_racc, mean_facc = np.mean(all_racc) * 100, np.mean(all_facc) * 100
            print(f"{'MEAN OVERALL':<25} | {mean_acc:5.2f}      | {mean_racc:5.2f}      | {mean_facc:5.2f}      | {mean_ap:5.2f}")
        print("=" * 70)

    print("\n🏁 所有动态鲁棒性测试执行完毕！")

if __name__ == '__main__':
    main()