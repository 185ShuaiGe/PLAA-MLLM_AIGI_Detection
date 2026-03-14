import os
import json
import torch
import numpy as np
import random
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from configs.path_config import PathConfig
from configs.model_config import ModelConfig
from utils.log_utils import Logger

class AIGIDataset(Dataset):
    """
    AI生成图像检测数据集类 (支持内存列表动态加载)
    """
    
    def __init__(
        self,
        path_config,
        model_config,
        data_list,            # 新增：接收切分好的数据列表
        base_img_dir,         # 新增：图片根目录路径
        split="train",
        image_size=224,
        use_augmentation=True
    ):
        self.path_config = path_config
        self.model_config = model_config
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation and split == "train"
        self.base_img_dir = base_img_dir
        
        self.logger = Logger(name=f"AIGIDataset_{split}")
        
        self.samples = data_list
        self.transform = self._build_transform()
        
        self.logger.info(f"Initialized {split} dataset with {len(self.samples)} samples")

    def _build_transform(self):
        transform_list = []
        
        if self.use_augmentation:
            transform_list.extend([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ])
        else:
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 拼接图片绝对路径
        # sample['images'][0] 形如 "./holmes_dataset/dataset_huggingface/1_fake/xxx.png"
        # lstrip 去掉前面的 "./" 避免拼接错误
        rel_img_path = sample['images'][0].lstrip("./") 
        full_img_path = os.path.join(self.base_img_dir, rel_img_path)
        
        # 2. 加载图片
        image = self._load_image(full_img_path)
        
        # 3. 获取纯净的二元标签
        label = sample.get("label", 0)
        
        annotation_info = {"image_path": full_img_path}
        text_prompt = sample.get("query", "<image>\nAnalyze this image and determine if it is real or AI-generated.")
        
        return image, label, annotation_info, text_prompt
    
    def _load_image(self, full_path):
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at: {os.path.abspath(full_path)}")
            
        image = Image.open(full_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor


def get_holmes_dataloaders(path_config, model_config, batch_size=32):
    """
    负责读取 clean-data.json 并在每次训练时动态划分 9:1
    """
    json_path = os.path.join(path_config.data_dir, "clean-data.json")
    base_img_dir = path_config.data_dir
    
    print(f"Loading dataset metadata from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        
    # 设定固定的随机种子，保证如果训练中断，重启时训练集/验证集的划分是一致的，防止数据穿越
    random.seed(42)
    random.shuffle(all_data)
    
    total_size = len(all_data)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    print("=" * 60)
    print("📊 Dataset Split Information (9:1 Dynamic Split)")
    print(f"  Total valid samples: {total_size}")
    print(f"  Training samples:    {train_size}")
    print(f"  Validation samples:  {val_size}")
    print("=" * 60)
    
    # 划分列表
    train_list = all_data[:train_size]
    val_list = all_data[train_size:]
    
    # 实例化数据集 (验证集关闭数据增强)
    train_dataset = AIGIDataset(
        path_config, model_config, 
        data_list=train_list, base_img_dir=base_img_dir, 
        split="train"
    )
    val_dataset = AIGIDataset(
        path_config, model_config, 
        data_list=val_list, base_img_dir=base_img_dir, 
        split="val", use_augmentation=False
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


# ========================== 下方完全保留你的原始代码，请勿修改 ==========================
class val_AIGIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        # 遍历 20 个类别子文件夹
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            # 遍历 0_real 和 1_fake
            for label_dir in ['0_real', '1_fake']:
                label_path = os.path.join(category_path, label_dir)
                if not os.path.isdir(label_path):
                    continue
                
                # 设定标签：0_real 为 0 (真实), 1_fake 为 1 (虚假)
                label = 0 if '0_real' in label_dir else 1
                
                # 获取所有图片
                for img_name in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # 补充空字典和默认的文本 prompt，凑齐 4 个返回值，防止 Validator 解包报错
        dummy_info = {"image_path": img_path}
        default_prompt = "<image>\nAnalyze this image and determine if it is real or AI-generated." 
            
        return image, label, dummy_info, default_prompt