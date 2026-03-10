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
from peft import LoraConfig, get_peft_model
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.plaa_mllm import PLAAMLLM
from data.dataset_loader import AIGIDataset
from utils.log_utils import Logger
from utils.device_utils import DeviceManager
from utils.metrics_utils import MetricsCalculator


class PLAAMLLMTrainer:
    """
    PLAA-MLLM 三阶段训练器
    """
    
    def __init__(
        self,
        model,
        model_config,
        device_config,
        path_config,
        stage=1
    ):
        self.model = model
        self.model_config = model_config
        self.device_config = device_config
        self.path_config = path_config
        self.stage = stage
        
        self.logger = Logger(name=f"Trainer_Stage{stage}")
        self.device_manager = DeviceManager(device_config)
        self.device = device_config.get_device()

        self.metrics_calculator = MetricsCalculator(path_config, mode='train', stage=self.stage)
        
        self.tokenizer = None
        self._init_tokenizer()
        
        self.reference_model = None
        if stage == 3:
            self._init_reference_model()
        
        self.best_metric = -float('inf')
        self.global_step = 0
        self.epoch = 0

        # 初始化 GradScaler
        # self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        self._setup_stage()
    
    def _init_reference_model(self):
        try:
            import copy
            self.reference_model = copy.deepcopy(self.model)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.logger.info("Reference model initialized for DPO training")
        except Exception as e:
            self.logger.warning(f"Failed to initialize reference model: {e}")
    
    def _init_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.llm_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer: {e}")
    
    def _setup_stage(self):
        self.logger.info(f"Setting up training stage {self.stage}")
        
        if self.stage == 1:
            self._freeze_clip_llm()
            self._unfreeze_artifact_adapter()
        elif self.stage == 2:
            self._freeze_visual_streams()
            self._apply_lora()
        elif self.stage == 3:
            self.logger.info("Stage 3: DPO training, keeping previous setup")
    
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
        self.logger.info("Unfreezing artifact stream and cross-attention adapter")
        
        if hasattr(self.model, 'dual_stream_encoder'):
            if hasattr(self.model.dual_stream_encoder, 'artifact_stream'):
                for param in self.model.dual_stream_encoder.artifact_stream.parameters():
                    param.requires_grad = True
        
        if hasattr(self.model, 'forensic_cross_attention'):
            for param in self.model.forensic_cross_attention.parameters():
                param.requires_grad = True
        
        # ==================== 【修复核心 1】显式解冻投影层和分类头 ====================
        for head in ['vision_token_proj', 'detection_head', 'mask_head']:
            if hasattr(self.model, head):
                for param in getattr(self.model, head).parameters():
                    param.requires_grad = True
        # =========================================================================
        
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters after unfreezing: {trainable_count:,}")
    
    def _freeze_visual_streams(self):
        self.logger.info("Freezing all visual streams")
        
        if hasattr(self.model, 'dual_stream_encoder'):
            for param in self.model.dual_stream_encoder.parameters():
                param.requires_grad = False
        
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters after freezing visual streams: {trainable_count:,}")
    
    def _apply_lora(self):
        self.logger.info(f"Applying LoRA with rank={self.model_config.lora_rank}")
        
        if hasattr(self.model, 'llm_infer') and hasattr(self.model.llm_infer, 'llm_model'):
            llm_model = self.model.llm_infer.llm_model
            
            if llm_model is not None:
                lora_config = LoraConfig(
                    r=self.model_config.lora_rank,
                    lora_alpha=self.model_config.lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                
                try:
                    self.model.llm_infer.llm_model = get_peft_model(llm_model, lora_config)
                    self.model.llm_infer.lora_config = lora_config
                    self.logger.info("LoRA applied successfully")
                    
                    trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    self.logger.info(f"Trainable parameters with LoRA: {trainable_count:,}")
                except Exception as e:
                    self.logger.warning(f"Failed to apply LoRA: {e}")
    
    def compute_loss_stage1(self, outputs, labels, masks=None):
        loss_dict = {}
        
        detection_logits = outputs.get('detection_logits', None)
        if detection_logits is not None:
            labels = labels.to(self.device).float().view(-1)                
            detection_logits = detection_logits.view(-1).to(torch.float32)
            bce_loss = F.binary_cross_entropy_with_logits(detection_logits, labels)
            loss_dict['bce_loss'] = bce_loss
        
        if masks is not None:
            pred_mask = outputs.get('pred_mask', None)
            if pred_mask is not None:
                dice_loss = self._dice_loss(pred_mask, masks.to(self.device))
                loss_dict['dice_loss'] = dice_loss
        
        if not loss_dict:
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict['dummy_loss'] = dummy_loss
        
        total_loss = sum(loss_dict.values())
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def _dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def compute_loss_stage2(self, outputs, labels, tokenizer=None, single_prompt=None):
        logits = outputs.get('logits', None)
        
        if logits is not None:
            vision_tokens = outputs.get('vision_tokens', None)
            num_vision_tokens = vision_tokens.size(1) if vision_tokens is not None else self.model_config.num_latent_queries
            
            expert_text = labels.get('expert_explanation', '')
            if tokenizer is not None and expert_text:
                if single_prompt is not None:
                    if isinstance(expert_text, (list, tuple)):
                        expert_text = expert_text[0]
                    full_text = single_prompt + " " + expert_text + tokenizer.eos_token
                else:
                    full_text = expert_text + tokenizer.eos_token
                
                tokenized_full = tokenizer(
                    full_text,
                    max_length=self.model_config.max_seq_len,
                    truncation=True,
                    return_tensors='pt'
                )
                full_target_ids = tokenized_full['input_ids'].to(self.device)
                
                num_prompt_tokens = 0
                if single_prompt is not None:
                    tokenized_prompt = tokenizer(
                        single_prompt,
                        return_tensors='pt'
                    )
                    num_prompt_tokens = tokenized_prompt['input_ids'].size(1)
                
                target_ids = full_target_ids.clone()
                if num_prompt_tokens > 0 and num_prompt_tokens < target_ids.size(1):
                    target_ids[0, :num_prompt_tokens] = -100
                
                ignore_vision = torch.full(
                    (target_ids.size(0), num_vision_tokens),
                    -100,
                    dtype=target_ids.dtype,
                    device=self.device
                )
                
                if ignore_vision.dim() != 2:
                    ignore_vision = ignore_vision.squeeze(0) if ignore_vision.dim() == 3 else ignore_vision
                if target_ids.dim() != 2:
                    target_ids = target_ids.squeeze(0) if target_ids.dim() == 3 else target_ids
                
                if ignore_vision.size(0) != target_ids.size(0):
                    if ignore_vision.size(0) == 1 and target_ids.size(0) > 1:
                        ignore_vision = ignore_vision.repeat(target_ids.size(0), 1)
                    elif target_ids.size(0) == 1 and ignore_vision.size(0) > 1:
                        target_ids = target_ids.repeat(ignore_vision.size(0), 1)
                
                target_mask = torch.cat([ignore_vision, target_ids], dim=1)
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_mask[..., 1:].contiguous().to(shift_logits.device)
                
                clm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                return {'clm_loss': clm_loss, 'total_loss': clm_loss}
        
        return {'total_loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
    
    def compute_loss_stage3(self, image, winner_text, loser_text, beta=0.1):
        try:
            policy_winner_log_prob = self._compute_log_prob(self.model, image, winner_text)
            policy_loser_log_prob = self._compute_log_prob(self.model, image, loser_text)
            
            if self.reference_model is not None:
                with torch.no_grad():
                    ref_winner_log_prob = self._compute_log_prob(self.reference_model, image, winner_text)
                    ref_loser_log_prob = self._compute_log_prob(self.reference_model, image, loser_text)
            else:
                ref_winner_log_prob = policy_winner_log_prob.detach()
                ref_loser_log_prob = policy_loser_log_prob.detach()
            
            policy_logratios = policy_winner_log_prob - policy_loser_log_prob
            ref_logratios = ref_winner_log_prob - ref_loser_log_prob
            
            logits = policy_logratios - ref_logratios
            dpo_loss = -F.logsigmoid(beta * logits).mean()
            
            return {'dpo_loss': dpo_loss, 'total_loss': dpo_loss}
        except Exception as e:
            self.logger.warning(f"Failed to compute DPO loss: {e}")
            return {'total_loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
    
    def _compute_log_prob(self, model, image, text):
        outputs = model(image, text)
        logits = outputs.get('logits')
        vision_tokens = outputs.get('vision_tokens')
        
        if logits is not None and self.tokenizer is not None:
            num_vision_tokens = vision_tokens.size(1) if vision_tokens is not None else self.model_config.num_latent_queries
            
            tokenized = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.model_config.max_seq_len
            ).to(self.device)
            
            input_ids = tokenized['input_ids']
            
            ignore_tokens = torch.full(
                (input_ids.size(0), num_vision_tokens),
                -100,
                dtype=input_ids.dtype,
                device=self.device
            )
            full_target_ids = torch.cat([ignore_tokens, input_ids], dim=1)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_target_ids[..., 1:].contiguous()
            
            valid_mask = (shift_labels != -100)
            safe_labels = shift_labels.clone()
            safe_labels[~valid_mask] = 0
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            if valid_mask.sum() > 0:
                log_probs = log_probs * valid_mask.float()
                return log_probs.sum() / valid_mask.sum()
            else:
                return torch.tensor(0.0, device=self.device)
        
        return torch.tensor(0.0, device=self.device)
    
    def train(self, train_loader, val_loader=None, num_epochs=3, learning_rate=1e-4, batch_size=4, checkpoint_path=None):
        self.logger.info(f"Starting training stage {self.stage}")
        
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
                
                if self.stage == 1 and len(true_labels) > 0 and len(pred_scores) > 0:
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
        
        # 1. 生成 Loss (和可能存在的 AUC) 的变化曲线图，Stage 2 也能正常出图
        self.metrics_calculator.plot_training_history(history, stage=self.stage)
            
        # 2. 仅在 Stage 1 存在分类指标数据时，才生成分类评估特定图表
        if self.stage == 1 and last_true_labels and last_pred_scores:
            self.metrics_calculator.visualize_metrics(last_metrics, last_true_labels, last_pred_scores)
        # =========================================================================
        
        return history

    def _train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss = 0.0
        
        # 【新增】设置梯度累加步数。相当于将真实的 Batch Size 放大 8 倍
        # 这将消除 batch_size=1 带来的极端梯度噪音
        accum_steps = getattr(self.model_config, 'grad_accum_steps', 8) 
        
        progress_bar = tqdm(loader, desc=f"Training Stage {self.stage}")
        optimizer.zero_grad() # 【修改】将清空梯度移到循环外
        
        for i, batch in enumerate(progress_bar):
            images, labels, annotation_info, text_prompts = batch
            images = images.to(self.device)
            labels_tensor = labels.to(self.device).float() if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=self.device).float()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.stage == 1:
                    outputs = self.model(images, text_prompts, text_guidance=text_prompts)
                    masks = annotation_info.get('mask') if isinstance(annotation_info, dict) else None
                    if masks is not None and isinstance(masks, torch.Tensor):
                        masks = masks.to(self.device)
                    loss_dict = self.compute_loss_stage1(outputs, labels_tensor, masks)
                    
                elif self.stage == 2:
                    single_prompt = text_prompts[0] if isinstance(text_prompts, (list, tuple)) else text_prompts
                    expert_text = annotation_info.get('expert_explanation', '')
                    if isinstance(expert_text, (list, tuple)):
                        expert_text = expert_text[0]
                    full_text = single_prompt + " " + expert_text + self.tokenizer.eos_token
                    
                    outputs = self.model(images, [full_text]) 
                    loss_dict = self.compute_loss_stage2(
                        outputs=outputs, 
                        labels=annotation_info, 
                        tokenizer=self.tokenizer,
                        single_prompt=single_prompt
                    )
                else:
                    loss_dict = {'total_loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
            
            loss = loss_dict['total_loss']

            # 【新增】将 loss 根据累加步数进行缩放
            loss = loss / accum_steps
            loss.backward()

            # 【新增】当达到累加步数，或者到了最后一个 batch 时，更新参数
            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 【修改】记录还原后的真实 loss
            actual_loss = loss.item() * accum_steps
            total_loss += actual_loss
            self.global_step += 1
            progress_bar.set_postfix({'loss': f"{actual_loss:.4f}"})
            
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
                
                # ==================== 【修复：添加 autocast 混合精度上下文】 ====================
                # 使用与训练相同的精度（如果报错仍有Half相关，可将bfloat16改为float16）
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16): 
                    if self.stage == 1:
                        # 整批推理
                        outputs = self.model(images, text_prompts, text_guidance=text_prompts)
                        
                        logits = outputs.get('detection_logits', None)
                        if logits is not None:
                            # 收集用于计算 AUC 的数据，注意要转回 float32 再存入 list
                            probs = torch.sigmoid(logits.float()).view(-1).cpu().tolist()
                            if isinstance(probs, float): probs = [probs]
                            all_pred_scores.extend(probs)
                            
                            lbls = labels_tensor.view(-1).cpu().tolist()
                            if isinstance(lbls, float): lbls = [lbls]
                            all_true_labels.extend(lbls)
                            
                        masks = annotation_info.get('mask') if isinstance(annotation_info, dict) else None
                        if masks is not None and isinstance(masks, torch.Tensor):
                            masks = masks.to(self.device)
                            
                        loss_dict = self.compute_loss_stage1(outputs, labels_tensor, masks)
                        loss = loss_dict['total_loss']
                        total_loss += loss.item()
                    elif self.stage == 2:
                        # 1. 提取单条 prompt
                        single_prompt = text_prompts[0] if isinstance(text_prompts, (list, tuple)) else text_prompts
                        
                        # 2. 提取专家解释文本 (请确保 'explanation' 是你 annotation_info 里存答案的 key)
                        expert_text = annotation_info.get('expert_explanation', '') # 如果你的 key 叫 'text' 或其他，请修改这里
                        if isinstance(expert_text, (list, tuple)):
                            expert_text = expert_text[0]
                            
                        # 3. 拼接完整的 Prompt + Answer
                        full_text = single_prompt + " " + expert_text + self.tokenizer.eos_token
                        
                        # 4. 【核心修复】将 full_text 传入模型！
                        # 必须传入完整文本，这样模型输出的 logits 长度才会和后面的 labels 长度(如 507)完全对齐
                        # 保持和原先 text_prompts 一致的数据类型（放到 list 中）
                        outputs = self.model(images, [full_text]) 
                        
                        # 5. 计算损失
                        loss_dict = self.compute_loss_stage2(
                            outputs=outputs, 
                            labels=annotation_info, 
                            tokenizer=self.tokenizer,
                            single_prompt=single_prompt
                        )
                        loss = loss_dict['total_loss']
                        total_loss += loss.item()
                        # =========================================================
                    else:
                        print("Stage 3 validation not implemented yet.")
                # =======================================================================
                    
        return total_loss / len(loader), all_true_labels, all_pred_scores 
    
    def _save_checkpoint(self, optimizer, scheduler, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': self.best_metric,
            'stage': self.stage
        }
        
        save_path = os.path.join(
            self.path_config.weights_dir,
            f'checkpoint_stage{self.stage}_latest.pt'
        )
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")
        
        if is_best:
            best_path = os.path.join(
                self.path_config.weights_dir,
                f'checkpoint_stage{self.stage}_best.pt'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best checkpoint saved to {best_path}")
    
    def _load_checkpoint(self, checkpoint_path, optimizer, scheduler):
        #先将权重加载到 CPU 上，避免 GPU 内存不足
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        #加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # 只有在“断点续训”相同阶段时，才加载优化器和进度。
        # 跨阶段（如 Stage 1 到 Stage 2）不应加载过去的优化器！
        ckpt_stage = checkpoint.get('stage', 1)
        if ckpt_stage == self.stage:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.epoch = checkpoint.get('epoch', 0)
                self.global_step = checkpoint.get('global_step', 0)
                self.best_metric = checkpoint.get('best_metric', -float('inf'))
                self.logger.info(f"Resuming Stage {self.stage} training from epoch {self.epoch}, step {self.global_step}")
            except Exception as e:
                self.logger.warning(f"Could not load optimizer state: {e}")
        else:
            self.logger.info(f"Transitioning from Stage {ckpt_stage} to Stage {self.stage}. Only model weights loaded. Optimizer reset.")
            
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

