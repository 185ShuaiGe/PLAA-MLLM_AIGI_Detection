import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 👇【新增这行】：导入 cudnn
import torch.backends.cudnn as cudnn 

# 👇【新增这行】：强制禁用 cuDNN，绕过 RTX40 系列显卡在混合精度下的 NaN Bug
cudnn.enabled = True

from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoTokenizer
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.ds_mome import DSMoME
from data.dataset_loader import AIGIDataset
from utils.log_utils import Logger
from utils.device_utils import DeviceManager
from utils.metrics_utils import MetricsCalculator


class DSMoMETrainer:
    """
    DS-MoME 二分类训练器
    """
    
    def __init__(
        self,
        model,
        model_config,
        device_config,
        path_config
    ):
        self.model = model
        self.model_config = model_config
        self.device_config = device_config
        self.path_config = path_config
        
        self.logger = Logger(name="Trainer")
        self.device_manager = DeviceManager(device_config)
        self.device = device_config.get_device()

        self.metrics_calculator = MetricsCalculator(path_config, mode='train')
        
        self.tokenizer = None
        self._init_tokenizer()
        
        self.best_metric = -float('inf')
        self.global_step = 0
        self.epoch = 0

        # 初始化 GradScaler
        # self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        self._setup_training()
    
    def _init_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.llm_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer: {e}")
    
    def _setup_training(self):
        self.logger.info("Setting up training")
        
        self._freeze_clip_llm()
        self._unfreeze_artifact_adapter()
    
    def _freeze_clip_llm(self):
        self.logger.info("Freezing CLIP semantic stream and LLM backbone")
        
        if hasattr(self.model, 'dual_stream_encoder'):
            if hasattr(self.model.dual_stream_encoder, 'semantic_stream'):
                for param in self.model.dual_stream_encoder.semantic_stream.parameters():
                    param.requires_grad = False
        
        if hasattr(self.model, 'llm_infer'):
                # 增加对 llm_model 是否为 None 的检查
                if hasattr(self.model.llm_infer, 'llm_model') and self.model.llm_infer.llm_model is not None:
                    for param in self.model.llm_infer.llm_model.parameters():
                        param.requires_grad = False
                else:
                    self.logger.warning("llm_model is None, skipping freeze.")
        
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters after freezing: {trainable_count:,}")
    
    def _unfreeze_artifact_adapter(self):
        self.logger.info("Unfreezing artifact stream and MoME fusion network")
        
        if hasattr(self.model, 'dual_stream_encoder'):
            if hasattr(self.model.dual_stream_encoder, 'artifact_stream'):
                for param in self.model.dual_stream_encoder.artifact_stream.parameters():
                    param.requires_grad = True
        
        if hasattr(self.model, 'mome_fusion'):
            for param in self.model.mome_fusion.parameters():
                param.requires_grad = True
        
        # 解冻视觉 token 投影层
        if hasattr(self.model, 'vision_token_proj'):
            for param in self.model.vision_token_proj.parameters():
                param.requires_grad = True
        
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters after unfreezing: {trainable_count:,}")
    
    def compute_loss(self, outputs, labels, text_input_ids=None, text_attention_mask=None):
        loss_dict = {}
        total_loss = 0.0
        
        # 计算检测损失
        detection_logits = outputs.get('detection_logits', None)
        if detection_logits is not None:
            labels = labels.to(self.device).float().view(-1)                
            detection_logits = detection_logits.view(-1).to(torch.float32)
            bce_loss = F.binary_cross_entropy_with_logits(detection_logits, labels)
            loss_dict['bce_loss'] = bce_loss
            total_loss += bce_loss
        
        # 计算 CLM Loss（仅当 enable_text_loss 为 True 时）
        if hasattr(self.model_config, 'enable_text_loss') and self.model_config.enable_text_loss:
            llm_logits = outputs.get('logits', None)
            if llm_logits is not None and text_input_ids is not None:
                # CLM Loss: 预测下一个 token
                # 将 logits 和 labels 移动到同一设备
                llm_logits = llm_logits.to(self.device)
                text_input_ids = text_input_ids.to(self.device)
                
                # Shift logits and labels for next-token prediction
                shift_logits = llm_logits[..., :-1, :].contiguous()
                shift_labels = text_input_ids[..., 1:].contiguous()
                
                # Compute loss
                clm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id if self.tokenizer is not None else -100
                )
                loss_dict['clm_loss'] = clm_loss
                total_loss += clm_loss
        
        loss_dict['total_loss'] = total_loss if total_loss != 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss_dict
    
    def train(self, train_loader, val_loader=None, num_epochs=3, learning_rate=1e-4, batch_size=4, checkpoint_path=None):
        self.logger.info("Starting training")
        
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path, optimizer, scheduler)
        
        # 用于保存每个 Epoch 数据的历史字典
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
        # 用于记录最后一次的验证结果，供给 ROC/PR 曲线绘图
        last_metrics = {}
        last_true_labels = []
        last_pred_scores = []
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 1. 训练一轮
            train_loss = self._train_epoch(train_loader, optimizer)
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            history['train_loss'].append(train_loss)
            
            # 2. 验证一轮
            if val_loader is not None:
                val_loss, true_labels, pred_scores = self._validate_epoch(val_loader)
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                history['val_loss'].append(val_loss)
                
                if len(true_labels) > 0 and len(pred_scores) > 0:
                    metrics = self.metrics_calculator.calculate_all_metrics(true_labels, pred_scores)
                    val_auc = metrics.get('auc_roc', 0.0)
                    self.logger.info(f"Val AUC: {val_auc:.4f}")
                    
                    history['val_auc'].append(val_auc)
                    # 缓存最后一轮的数据用于出图
                    last_metrics = metrics
                    last_true_labels = true_labels
                    last_pred_scores = pred_scores
                else:
                    history['val_auc'].append(0.0)
                
                # 保存最佳模型
                if val_loss < self.best_metric or self.best_metric == -float('inf'):
                    self.best_metric = val_loss
                    self._save_checkpoint(optimizer, scheduler, is_best=True)
            
            scheduler.step()
            self._save_checkpoint(optimizer, scheduler, is_best=False)
        
        # ==================== 【利用 metrics_utils 生成可视化】 ====================
        self.logger.info("Training completed. Generating all visualizations...")
        
        # 1. 生成 Loss 和 AUC 的变化曲线图
        self.metrics_calculator.plot_training_history(history)
            
        # 2. 生成分类评估特定图表
        if last_true_labels and last_pred_scores:
            self.metrics_calculator.visualize_metrics(last_metrics, last_true_labels, last_pred_scores)
        # =========================================================================
        
        return history

    def _train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss = 0.0
        
        # 设置梯度累加步数
        accum_steps = getattr(self.model_config, 'grad_accum_steps', 8) 
        
        progress_bar = tqdm(loader, desc="Training")
        optimizer.zero_grad()
        
        for i, batch in enumerate(progress_bar):
            images, labels, annotation_info, text_prompts = batch
            images = images.to(self.device)
            labels_tensor = labels.to(self.device).float() if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=self.device).float()
            
            # Tokenize text prompts if needed for CLM loss
            text_input_ids = None
            text_attention_mask = None
            if hasattr(self.model_config, 'enable_text_loss') and self.model_config.enable_text_loss and self.tokenizer is not None:
                tokenized = self.tokenizer(
                    text_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.model_config.max_seq_len
                )
                text_input_ids = tokenized['input_ids']
                text_attention_mask = tokenized['attention_mask']
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(images, text_prompts)
                loss_dict = self.compute_loss(outputs, labels_tensor, text_input_ids, text_attention_mask)
            
            loss = loss_dict['total_loss']

            # 将 loss 根据累加步数进行缩放
            loss = loss / accum_steps
            loss.backward()

            # 当达到累加步数，或者到了最后一个 batch 时，更新参数
            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录还原后的真实 loss
            actual_loss = loss.item() * accum_steps
            total_loss += actual_loss
            self.global_step += 1
            # 更新进度条显示更多信息
            postfix = {'loss': f"{actual_loss:.4f}"}
            if 'clm_loss' in loss_dict:
                postfix['clm'] = f"{loss_dict['clm_loss'].item():.4f}"
            progress_bar.set_postfix(postfix)
            
        return total_loss / len(loader)
    

    def _validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_true_labels = []
        all_pred_scores = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                images, labels, annotation_info, text_prompts = batch
                images = images.to(self.device)
                labels_tensor = labels.to(self.device).float() if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=self.device).float()
                
                # 使用与训练相同的精度
                with torch.autocast(device_type='cuda', dtype=torch.float16): 
                    outputs = self.model(images, text_prompts)
                    logits = outputs.get('detection_logits', None)
                    
                    if logits is not None:
                        # 收集用于计算 AUC 的数据，注意要转回 float32 再存入 list
                        probs = torch.sigmoid(logits.float()).view(-1).cpu().tolist()
                        if isinstance(probs, float): probs = [probs]
                        all_pred_scores.extend(probs)
                        
                        lbls = labels_tensor.view(-1).cpu().tolist()
                        if isinstance(lbls, float): lbls = [lbls]
                        all_true_labels.extend(lbls)
                        
                    loss_dict = self.compute_loss(outputs, labels_tensor)
                    loss = loss_dict['total_loss']
                    total_loss += loss.item()
                    
        return total_loss / len(loader), all_true_labels, all_pred_scores 
    
    def _save_checkpoint(self, optimizer, scheduler, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': self.best_metric
        }
        
        save_path = os.path.join(
            self.path_config.weights_dir,
            'checkpoint_latest.pt'
        )
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")
        
        if is_best:
            best_path = os.path.join(
                self.path_config.weights_dir,
                'checkpoint_best.pt'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best checkpoint saved to {best_path}")
    
    def _load_checkpoint(self, checkpoint_path, optimizer, scheduler):
        #先将权重加载到 CPU 上，避免 GPU 内存不足
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        #加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_metric = checkpoint.get('best_metric', -float('inf'))
            self.logger.info(f"Resuming training from epoch {self.epoch}, step {self.global_step}")
        except Exception as e:
            self.logger.warning(f"Could not load optimizer state: {e}")
            
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
