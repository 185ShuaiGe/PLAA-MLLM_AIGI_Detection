
import os
import json
import torch
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.plaa_mllm import PLAAMLLM
from data.dataset_loader import AIGIDataset
from utils.log_utils import Logger


class PLAAMLLMValidator:
    """
    PLAA-MLLM 模型验证器
    """
    
    def __init__(
        self,
        model: PLAAMLLM,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        path_config: PathConfig
    ):
        """
        初始化验证器
        
        Args:
            model: PLAA-MLLM 模型
            model_config: 模型配置
            device_config: 设备配置
            path_config: 路径配置
        """
        self.model = model
        self.model_config = model_config
        self.device_config = device_config
        self.path_config = path_config
        
        self.logger = Logger(name="Validator")
        self.device = device_config.get_device()
        
        self.results = []
    
    def validate(
        self,
        val_loader: DataLoader,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行验证
        
        Args:
            val_loader: 验证数据加载器
            save_results: 是否保存结果
            output_dir: 输出目录
        
        Returns:
            Dict: 验证结果字典
        """
        self.logger.info("Starting validation")
        self.model.eval()
        self.results = []
        
        output_dir = output_dir or self.path_config.outputs_dir
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                images, labels, annotation_info, text_prompts = batch
                images = images.to(self.device)
                
                for i in range(len(images)):
                    single_image = images[i:i+1]
                    single_label = labels[i] if isinstance(labels, torch.Tensor) else labels[i]
                    single_info = {}
                    for k, v in annotation_info.items():
                        val = v[i] if isinstance(v, (list, tuple, torch.Tensor)) else v
                        # 如果提取出来的是 Tensor，将其转换为普通的 Python 数据类型（int/float/list）
                        if isinstance(val, torch.Tensor):
                            val = val.item() if val.numel() == 1 else val.cpu().tolist()
                        single_info[k] = val                    
                        single_prompt = text_prompts[i] if isinstance(text_prompts, (list, tuple)) else text_prompts                    
                    result = self._validate_single(single_image, single_label, single_info, single_prompt)
                    self.results.append(result)
        
        # if save_results:
        #     self._save_results(output_dir)
        
        return self._aggregate_results()
    
    def _validate_single(
        self,
        image: torch.Tensor,
        label: Any,
        annotation_info: Dict[str, Any],
        text_prompt: str
    ) -> Dict[str, Any]:
        """
        验证单个样本
        
        Args:
            image: 输入图像
            label: 真实标签
            annotation_info: 标注信息
            text_prompt: 文本提示
        
        Returns:
            Dict: 单个样本验证结果
        """
        outputs = self.model(image, text_prompt)
        
        detection_logits = outputs.get('detection_logits', None)
        if detection_logits is not None:
            detection_result = torch.sigmoid(detection_logits).item()
        else:
            detection_result = 0.5
        
        pred_mask = outputs.get('pred_mask', None)
        explanation = outputs.get('explanation', '')
        
        result = {
            'image_path': annotation_info.get('image_path', ''),
            'true_label': int(label) if isinstance(label, torch.Tensor) else label,
            'pred_score': float(detection_result) if isinstance(detection_result, torch.Tensor) else detection_result,
            'explanation': explanation,
            'annotation': annotation_info
        }
        
        if pred_mask is not None:
            result['pred_mask'] = pred_mask.cpu().numpy().tolist() if isinstance(pred_mask, torch.Tensor) else pred_mask
        
        return result
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """
        汇总验证结果
        
        Returns:
            Dict: 汇总结果
        """
        if not self.results:
            return {}
        
        true_labels = [r['true_label'] for r in self.results]
        pred_scores = [r['pred_score'] for r in self.results]
        
        aggregate = {
            'total_samples': len(self.results),
            'true_labels': true_labels,
            'pred_scores': pred_scores,
            'results': self.results
        }
        
        return aggregate
    
    def _save_results(self, output_dir: str) -> None:
        """
        保存验证结果为 JSON
        
        Args:
            output_dir: 输出目录
        """
        output_file = os.path.join(output_dir, 'validation_results.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'aggregate': self._aggregate_results(),
                'individual_results': self.results
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Validation results saved to {output_file}")
