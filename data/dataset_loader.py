
import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset
from torchvision import transforms
from configs.path_config import PathConfig
from configs.model_config import ModelConfig
from utils.log_utils import Logger


class AIGIDataset(Dataset):
    """
    AI生成图像检测数据集类
    支持三种训练阶段的数据形态
    """
    
    def __init__(
        self,
        path_config,
        model_config,
        stage=1,
        split="train",
        image_size=224,
        use_augmentation=True
    ):
        self.path_config = path_config
        self.model_config = model_config
        self.stage = stage
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation and split == "train"
        
        self.logger = Logger(name=f"AIGIDataset_{split}_stage{stage}")
        
        self.data_dir = os.path.join(path_config.data_dir, split)
        self.annotation_file = os.path.join(self.data_dir, "annotations.json")
        
        self.samples = self._load_annotations()
        self.transform = self._build_transform()
        
        self.logger.info(f"Loaded {len(self.samples)} samples for {split} split (stage {stage})")
    
    def _load_annotations(self):
        samples = []
        
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                samples = annotations.get("samples", [])
            except Exception as e:
                self.logger.warning(f"Failed to load annotations: {e}")
        
        if not samples:
            self.logger.warning("No annotations found, using dummy data")
            samples = self._generate_dummy_samples()
        
        return samples
    
    def _generate_dummy_samples(self):
        dummy_samples = []
        
        for i in range(10):
            sample = {
                "image_path": f"dummy_{i}.jpg",
                "label": i % 2,
                "stage": self.stage
            }
            
            if self.stage >= 2:
                sample.update({
                    "text_query": "Is this image AI-generated?",
                    "mask": None,
                    "expert_explanation": "This image contains typical AI artifacts."
                })
            
            if self.stage >= 3:
                sample.update({
                    "winner": "Yes, this image appears to be AI-generated due to inconsistent textures.",
                    "loser": "Maybe real?"
                })
            
            dummy_samples.append(sample)
        
        return dummy_samples
    
    def _build_transform(self):
        transform_list = []
        
        if self.use_augmentation:
            transform_list.extend([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
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
    
    def _build_mask_transform(self):
        transform_list = []
        
        if self.use_augmentation:
            transform_list.extend([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
        else:
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        transform_list.append(transforms.ToTensor())
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = self._load_image(sample.get("image_path", ""))
        except Exception as e:
            self.logger.error(f"Failed to load image at index {idx}: {e}")
            image = torch.zeros(3, self.image_size, self.image_size)
        
        label = sample.get("label", 0)
        annotation_info = self._extract_annotation_info(sample)
        text_prompt = self._get_text_prompt(sample)
        
        return image, label, annotation_info, text_prompt
    
    def _load_image(self, image_path):
        full_path = os.path.join(self.data_dir, image_path)
        
        if os.path.exists(full_path):
            image = Image.open(full_path).convert("RGB")
        else:
            image = Image.fromarray(np.uint8(np.random.randint(0, 255, (256, 256, 3))))
        
        image_tensor = self.transform(image)
        return image_tensor
    
    def _load_mask(self, mask_path_or_data):
        if mask_path_or_data is None:
            return None
        
        try:
            if isinstance(mask_path_or_data, str):
                full_path = os.path.join(self.data_dir, mask_path_or_data)
                if os.path.exists(full_path):
                    mask = Image.open(full_path).convert("L")
                else:
                    mask = Image.fromarray(np.uint8(np.zeros((256, 256))))
            elif isinstance(mask_path_or_data, np.ndarray):
                mask = Image.fromarray(mask_path_or_data.astype(np.uint8))
            else:
                mask = Image.fromarray(np.uint8(np.zeros((256, 256))))
            
            mask_transform = self._build_mask_transform()
            mask_tensor = mask_transform(mask)
            
            return mask_tensor
        except Exception as e:
            self.logger.warning(f"Failed to load mask: {e}")
            return None
    
    def _extract_annotation_info(self, sample):
        info = {
            "image_path": sample.get("image_path", ""),
            "stage": sample.get("stage", self.stage)
        }
        
        if self.stage >= 2:
            mask_data = sample.get("mask")
            mask_tensor = self._load_mask(mask_data)
            
            info.update({
                "text_query": sample.get("text_query", ""),
                "expert_explanation": sample.get("expert_explanation", ""),
                "mask": mask_tensor
            })
        
        if self.stage >= 3:
            info.update({
                "winner": sample.get("winner", ""),
                "loser": sample.get("loser", "")
            })
        
        return info
    
    def _get_text_prompt(self, sample):
        if self.stage == 1:
            return "Classify this image as real or AI-generated."
        elif self.stage == 2:
            return sample.get("text_query", "Analyze this image.")
        else:
            return sample.get("text_query", "Which answer is better?")

