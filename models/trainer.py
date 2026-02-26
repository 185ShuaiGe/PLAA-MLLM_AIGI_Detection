
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.tokenizer = None
        self._init_tokenizer()
        
        self.reference_model = None
        if stage == 3:
            self._init_reference_model()
        
        self.best_metric = -float('inf')
        self.global_step = 0
        self.epoch = 0
        
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
            labels = labels.to(self.device).float()
            if detection_logits.dim() > 1:
                detection_logits = detection_logits.view(-1)
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
                    full_text = single_prompt + " " + expert_text
                else:
                    full_text = expert_text
                
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
                shift_labels = target_mask[..., 1:].contiguous()
                
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
    
    def train(self, train_loader, val_loader=None, num_epochs=10, learning_rate=1e-4, batch_size=8, checkpoint_path=None):
        self.logger.info(f"Starting training stage {self.stage}")
        
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path, optimizer, scheduler)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            train_loss = self._train_epoch(train_loader, optimizer)
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                
                if val_loss < self.best_metric or self.best_metric == -float('inf'):
                    self.best_metric = val_loss
                    self._save_checkpoint(optimizer, scheduler, is_best=True)
            
            scheduler.step()
            self._save_checkpoint(optimizer, scheduler, is_best=False)
    
    def _train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(loader, desc=f"Training Stage {self.stage}")
        
        for batch in progress_bar:
            images, labels, annotation_info, text_prompts = batch
            images = images.to(self.device)
            
            optimizer.zero_grad()
            
            batch_size_local = images.size(0)
            batch_outputs = []
            batch_losses = []
            
            for i in range(batch_size_local):
                single_image = images[i:i+1]
                single_prompt = text_prompts[i] if isinstance(text_prompts, (list, tuple)) else text_prompts
                
                if self.stage == 1:
                    outputs = self.model(single_image, single_prompt)
                    single_label = labels[i:i+1] if isinstance(labels, torch.Tensor) else [labels[i]]
                    single_label_tensor = torch.tensor([single_label], device=self.device) if not isinstance(labels, torch.Tensor) else single_label
                    
                    mask = None
                    if isinstance(annotation_info, dict):
                        mask_data = annotation_info.get('mask')
                        if isinstance(mask_data, (list, tuple)) and i < len(mask_data):
                            mask = mask_data[i]
                    loss_dict = self.compute_loss_stage1(outputs, single_label_tensor, mask)
                elif self.stage == 2:
                    single_info = {k: v[i] if isinstance(v, (list, tuple, torch.Tensor)) else v for k, v in annotation_info.items()}
                    expert_explanation = single_info.get('expert_explanation', '')
                    combined_text = single_prompt + " " + expert_explanation if expert_explanation else single_prompt
                    outputs = self.model(single_image, combined_text)
                    loss_dict = self.compute_loss_stage2(outputs, single_info, self.tokenizer, single_prompt)
                elif self.stage == 3:
                    single_info = {k: v[i] if isinstance(v, (list, tuple, torch.Tensor)) else v for k, v in annotation_info.items()}
                    winner_text = single_info.get('winner', '')
                    loser_text = single_info.get('loser', '')
                    loss_dict = self.compute_loss_stage3(single_image, winner_text, loser_text)
                
                batch_losses.append(loss_dict['total_loss'])
            
            loss = torch.mean(torch.stack(batch_losses))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(loader)
    
    def _validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                images, labels, annotation_info, text_prompts = batch
                images = images.to(self.device)
                
                batch_size_local = images.size(0)
                batch_losses = []
                
                for i in range(batch_size_local):
                    single_image = images[i:i+1]
                    single_prompt = text_prompts[i] if isinstance(text_prompts, (list, tuple)) else text_prompts
                    
                    
                    if self.stage == 1:
                        outputs = self.model(single_image, single_prompt)
                        single_label = labels[i:i+1] if isinstance(labels, torch.Tensor) else [labels[i]]
                        single_label_tensor = torch.tensor([single_label], device=self.device) if not isinstance(labels, torch.Tensor) else single_label
                        
                        mask = None
                        if isinstance(annotation_info, dict):
                            mask_data = annotation_info.get('mask')
                            if isinstance(mask_data, (list, tuple)) and i < len(mask_data):
                                mask = mask_data[i]
                        loss_dict = self.compute_loss_stage1(outputs, single_label_tensor, mask)
                    elif self.stage == 2:
                        single_info = {k: v[i] if isinstance(v, (list, tuple, torch.Tensor)) else v for k, v in annotation_info.items()}
                        expert_explanation = single_info.get('expert_explanation', '')
                        combined_text = single_prompt + " " + expert_explanation if expert_explanation else single_prompt
                        outputs = self.model(single_image, combined_text)
                        loss_dict = self.compute_loss_stage2(outputs, single_info, self.tokenizer, single_prompt)
                    else:
                        single_info = {k: v[i] if isinstance(v, (list, tuple, torch.Tensor)) else v for k, v in annotation_info.items()}
                        winner_text = single_info.get('winner', '')
                        loser_text = single_info.get('loser', '')
                        loss_dict = self.compute_loss_stage3(single_image, winner_text, loser_text)
                    
                    batch_losses.append(loss_dict['total_loss'])
                
                loss = torch.mean(torch.stack(batch_losses))
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', -float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")

