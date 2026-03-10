import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import CLIPModel
from tqdm import tqdm
import warnings

# 忽略烦人的警告
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 配置参数 (根据你的描述已填好)
# ---------------------------------------------------------
# 你的数据集根目录 (fdmas/test)
TEST_ROOT = '/data/Disk_A/wangxinchang/Datasets/fdmas/test/'

# 你的权重路径 
MODEL_PATH = './checkpoints/train_ProGAN__2026_01_29_11_55_35/10000_steps_latest_net_Model.pth'
#MODEL_PATH = './checkpoints/C2P_CLIP_release_20240901/last_model.pth'

# CLIP 预训练模型路径
CLIP_PATH = './pretrained/clip-vit-large-patch14'

# 批大小 (显存不够可调小)
BATCH_SIZE = 32
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------
# 2. 定义模型结构 (与 inference.py 保持一致)
# ---------------------------------------------------------
class C2P_CLIP(nn.Module):
    def __init__(self, name, num_classes=1):
        super(C2P_CLIP, self).__init__()
        # 加载本地 CLIP
        self.model = CLIPModel.from_pretrained(name)
        # 推理阶段不需要文本分支，删除以节省显存
        del self.model.text_model
        del self.model.text_projection
        del self.model.logit_scale
        
        self.model.vision_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        # 定义分类头
        self.model.fc = nn.Linear(768, num_classes)

    def encode_image(self, img):
        vision_outputs = self.model.vision_model(pixel_values=img)
        pooled_output = vision_outputs[1] if isinstance(vision_outputs, tuple) else vision_outputs.pooler_output # pooled_output
        image_features = self.model.visual_projection(pooled_output)
        return image_features    

    def forward(self, img):
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return self.model.fc(image_embeds)

# ---------------------------------------------------------
# 3. 数据预处理
# ---------------------------------------------------------
def get_transforms():
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        from PIL import Image
        BICUBIC = Image.BICUBIC
        
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
    ])

# ---------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------
def main():
    print(f"🚀 开始测试 FDMAS 数据集...")
    print(f"📂 数据集路径: {TEST_ROOT}")
    print(f"⚖️  模型路径: {MODEL_PATH}")
    
    # 1. 初始化模型
    model = C2P_CLIP(name=CLIP_PATH, num_classes=1)
    
    # 2. 加载权重
    print("⏳ 正在加载权重...")
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # 处理可能的权重键值不匹配 (去除 module. 前缀)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print("✅ 模型加载完成！")

    # 3. 获取所有子数据集文件夹
    sub_datasets = sorted([d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))])
    
    # 4. 打印表格头
    print("\n" + "="*55)
    print(f"{'Dataset':<25} | {'ACC (%)':<10} | {'AP (%)':<10}")
    print("-" * 55)

    all_acc = []
    all_ap = []

    # 5. 循环测试每个子数据集
    transform = get_transforms()
    
    for dataset_name in sub_datasets:
        dataset_path = os.path.join(TEST_ROOT, dataset_name)
        
        # 使用 ImageFolder 自动识别 0_real 和 1_fake
        # ImageFolder 会自动按字母排序：
        # '0_real' -> class 0 (Real)
        # '1_fake' -> class 1 (Fake)
        try:
            dataset = ImageFolder(root=dataset_path, transform=transform)
        except Exception as e:
            print(f"{dataset_name:<25} | Error loading: {e}")
            continue

        # 验证类别映射是否正确
        # 我们期望 {'0_real': 0, '1_fake': 1}
        class_to_idx = dataset.class_to_idx
        # 简单检查：确保 1_fake 对应的 index 是 1，或者 fake 对应 1
        fake_label = 1
        if '1_fake' in class_to_idx and class_to_idx['1_fake'] != 1:
             # 如果顺序反了，需要标记一下（虽然 0_xx 肯定排在 1_xx 前面，通常不会错）
             print(f"Warning: Class mapping unexpected for {dataset_name}: {class_to_idx}")

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        y_true = []
        y_pred = [] # 存储概率值
        
        # 推理循环
        with torch.no_grad():
            for images, labels in dataloader: # tqdm(dataloader, leave=False, desc=dataset_name):
                images = images.to(DEVICE)
                
                # 前向传播
                output = model(images) 
                
                # 计算概率 (Sigmoid)
                probs = torch.sigmoid(output).squeeze()
                
                # 处理 batch_size=1 的情况
                if probs.ndim == 0:
                    probs = probs.unsqueeze(0)

                y_pred.extend(probs.cpu().tolist())
                y_true.extend(labels.tolist())

        # 计算指标
        y_true = np.array(y_true)
        y_pred = np.array(y_pred).reshape(-1, 1)
        
        # 计算 ACC (阈值 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        # 计算 AP
        ap = average_precision_score(y_true, y_pred)
        
        all_acc.append(acc)
        all_ap.append(ap)
        
        # 打印这一行的结果
        print(f"{dataset_name:<25} | {acc*100:5.2f}      | {ap*100:5.2f}")

    # 6. 打印平均值
    print("-" * 55)
    mean_acc = np.mean(all_acc) * 100
    mean_ap = np.mean(all_ap) * 100
    print(f"{'MEAN':<25} | {mean_acc:5.2f}      | {mean_ap:5.2f}")
    print("=" * 55)

if __name__ == '__main__':
    main()