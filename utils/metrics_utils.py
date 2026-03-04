
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, average_precision_score
from configs.path_config import PathConfig
from utils.log_utils import Logger


class MetricsCalculator:
    """
    多维度指标计算器
    """
    
    def __init__(self, path_config: PathConfig):
        """
        初始化指标计算器
        
        Args:
            path_config: 路径配置
        """
        self.path_config = path_config
        self.logger = Logger(name="MetricsCalculator")
        
        self.metrics_dir = os.path.join(path_config.outputs_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def calculate_all_metrics(
        self,
        true_labels: List[int],
        pred_scores: List[float],
        pred_masks: Optional[List[np.ndarray]] = None,
        true_masks: Optional[List[np.ndarray]] = None,
        explanations: Optional[List[str]] = None,
        references: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            true_labels: 真实标签列表
            pred_scores: 预测分数列表
            pred_masks: 预测掩码列表
            true_masks: 真实掩码列表
            explanations: 生成的解释列表
            references: 参考解释列表
        
        Returns:
            Dict: 指标字典
        """
        metrics = {}
        
        metrics.update(self.calculate_detection_metrics(true_labels, pred_scores))
        
        if pred_masks is not None and true_masks is not None:
            metrics.update(self.calculate_localization_metrics(pred_masks, true_masks))
        
        if explanations is not None and references is not None:
            metrics.update(self.calculate_text_metrics(explanations, references))
        
        self.logger.info(f"Calculated metrics: {metrics}")
        return metrics
    
    def calculate_detection_metrics(
        self,
        true_labels: List[int],
        pred_scores: List[float]
    ) -> Dict[str, float]:
        """
        计算检测指标：AUC-ROC, EER, F1-Score
        
        Args:
            true_labels: 真实标签列表
            pred_scores: 预测分数列表
        
        Returns:
            Dict: 检测指标字典
        """
        metrics = {}
        
        if len(set(true_labels)) > 1:
            metrics['auc_roc'] = roc_auc_score(true_labels, pred_scores)
            
            fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            metrics['eer'] = (fpr[eer_idx] + fnr[eer_idx]) / 2
            
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            pred_labels = [1 if s >= best_threshold else 0 for s in pred_scores]
            metrics['f1_score'] = f1_score(true_labels, pred_labels)
            
            metrics['ap'] = average_precision_score(true_labels, pred_scores)
        
        return metrics
    
    def calculate_localization_metrics(
        self,
        pred_masks: List[np.ndarray],
        true_masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        计算定位指标：mAP, IoU
        
        Args:
            pred_masks: 预测掩码列表
            true_masks: 真实掩码列表
        
        Returns:
            Dict: 定位指标字典
        """
        metrics = {}
        ious = []
        
        for pred_mask, true_mask in zip(pred_masks, true_masks):
            if pred_mask is not None and true_mask is not None:
                iou = self._calculate_iou(pred_mask, true_mask)
                ious.append(iou)
        
        if ious:
            metrics['mean_iou'] = np.mean(ious)
            metrics['map'] = np.mean([max(0, iou) for iou in ious])
        
        return metrics
    
    def _calculate_iou(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
        """
        计算 IoU
        
        Args:
            pred_mask: 预测掩码
            true_mask: 真实掩码
        
        Returns:
            float: IoU 值
        """
        pred_binary = (pred_mask >= 0.5).astype(np.float32)
        true_binary = (true_mask >= 0.5).astype(np.float32)
        
        intersection = np.sum(pred_binary * true_binary)
        union = np.sum(pred_binary) + np.sum(true_binary) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_text_metrics(
        self,
        explanations: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算文本指标：ROUGE-L, CIDEr
        预留 LLM-as-a-judge 接口
        
        Args:
            explanations: 生成的解释列表
            references: 参考解释列表
        
        Returns:
            Dict: 文本指标字典
        """
        metrics = {}
        
        metrics['rouge_l'] = self._calculate_rouge_l(explanations, references)
        metrics['cider'] = self._calculate_cider(explanations, references)
        
        return metrics
    
    def _calculate_rouge_l(self, candidates: List[str], references: List[str]) -> float:
        """
        计算 ROUGE-L
        
        Args:
            candidates: 候选文本列表
            references: 参考文本列表
        
        Returns:
            float: ROUGE-L 分数
        """
        scores = []
        for cand, ref in zip(candidates, references):
            scores.append(self._lcs_score(cand, ref))
        return np.mean(scores) if scores else 0.0
    
    def _lcs_score(self, candidate: str, reference: str) -> float:
        """
        计算 LCS 分数
        
        Args:
            candidate: 候选文本
            reference: 参考文本
        
        Returns:
            float: LCS 分数
        """
        cand_tokens = candidate.split()
        ref_tokens = reference.split()
        
        m, n = len(cand_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if cand_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_len = dp[m][n]
        if n == 0:
            return 0.0
        return lcs_len / n
    
    def _calculate_cider(self, candidates: List[str], references: List[str]) -> float:
        """
        计算 CIDEr 分数
        
        Args:
            candidates: 候选文本列表
            references: 参考文本列表
        
        Returns:
            float: CIDEr 分数
        """
        return 0.0
    
    def llm_as_judge(
        self,
        explanations: List[str],
        references: List[str],
        images: Optional[List] = None
    ) -> Dict[str, float]:
        """
        LLM-as-a-judge 接口（预留）
        
        Args:
            explanations: 生成的解释列表
            references: 参考解释列表
            images: 图像列表
        
        Returns:
            Dict: 评分结果
        """
        return {'factuality': 0.0, 'consistency': 0.0, 'overall': 0.0}
    
    def visualize_metrics(self, metrics: Dict[str, float], true_labels: List[int], pred_scores: List[float]) -> None:
        """
        可视化指标
        
        Args:
            metrics: 指标字典
            true_labels: 真实标签
            pred_scores: 预测分数
        """
        if len(set(true_labels)) > 1:
            self._plot_roc_curve(true_labels, pred_scores)
            self._plot_precision_recall_curve(true_labels, pred_scores)
        
        self._plot_metric_bar_chart(metrics)
    
    def _plot_roc_curve(self, true_labels: List[int], pred_scores: List[float]) -> None:
        """
        绘制 ROC 曲线
        """
        fpr, tpr, _ = roc_curve(true_labels, pred_scores)
        auc = roc_auc_score(true_labels, pred_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(self.metrics_dir, 'roc_curve.png'))
        plt.close()
    
    def _plot_precision_recall_curve(self, true_labels: List[int], pred_scores: List[float]) -> None:
        """
        绘制 Precision-Recall 曲线
        """
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
        ap = average_precision_score(true_labels, pred_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(self.metrics_dir, 'pr_curve.png'))
        plt.close()
    
    def _plot_metric_bar_chart(self, metrics: Dict[str, float]) -> None:
        """
        绘制指标柱状图
        """
        names = list(metrics.keys())
        values = list(metrics.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, values)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Metrics Summary')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'metrics_bar_chart.png'))
        plt.close()

    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        绘制训练损失和验证指标随 Epoch 变化的曲线
        
        Args:
            history: 包含 'train_loss', 'val_loss', 'val_auc' 等列表的字典
        """
        if not history or not history.get('train_loss'):
            return
            
        epochs = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))
        
        # 绘制 Loss 曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='s')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(epochs)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # 绘制 AUC 曲线
        plt.subplot(1, 2, 2)
        if 'val_auc' in history and history['val_auc']:
            plt.plot(epochs, history['val_auc'], label='Validation AUC', marker='^', color='green')
            plt.title('Validation AUC over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('AUC Score')
            plt.xticks(epochs)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
        plt.tight_layout()
        save_path = os.path.join(self.metrics_dir, 'training_history_curve.png')
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Training history curve saved to {save_path}")