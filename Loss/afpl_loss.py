"""
Loss functions for AFPL-Net

Implements three loss components:
1. Focal Loss for classification (lane/non-lane)
2. BCE Loss for centerness (point quality)
3. Smooth L1 Loss for polar regression (θ, r)
"""

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, 1, H, W] or [B, H, W]
            target: Ground truth [B, H, W]
            
        Returns:
            Focal loss value
        """
        # Flatten tensors
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # [B, H, W]
        pred = pred.reshape(-1)  # [B*H*W]
        target = target.reshape(-1)  # [B*H*W]
        
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Compute focal term: (1 - p_t)^gamma
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute alpha term
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal loss = alpha * focal_term * BCE
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CenternessLoss(nn.Module):
    """
    Binary Cross Entropy Loss for centerness prediction
    
    Centerness measures how close a point is to the lane centerline.
    Only computed for positive samples (lane pixels).
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted centerness logits [B, 1, H, W] or [B, H, W]
            target: Ground truth centerness [B, H, W]
            mask: Optional mask for positive samples [B, H, W]
            
        Returns:
            BCE loss value
        """
        # Flatten tensors
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # [B, H, W]
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        # Apply mask if provided (only compute loss on positive samples)
        if mask is not None:
            mask = mask.reshape(-1)
            pred = pred[mask > 0]
            target = target[mask > 0]
            
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Binary cross entropy
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction=self.reduction)
        
        return loss


class PolarRegressionLoss(nn.Module):
    """
    Smooth L1 Loss for polar coordinate regression
    
    Predicts (θ, r) for each lane point relative to global pole.
    Only computed for positive samples (lane pixels).
    """
    
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, theta_pred, r_pred, theta_target, r_target, mask):
        """
        Args:
            theta_pred: Predicted angle [B, 1, H, W]
            r_pred: Predicted radius [B, 1, H, W]
            theta_target: Target angle [B, H, W]
            r_target: Target radius [B, H, W]
            mask: Positive sample mask [B, H, W]
            
        Returns:
            Smooth L1 loss value
        """
        # Flatten tensors
        theta_pred = theta_pred.squeeze(1).reshape(-1)
        r_pred = r_pred.squeeze(1).reshape(-1)
        theta_target = theta_target.reshape(-1)
        r_target = r_target.reshape(-1)
        mask = mask.reshape(-1)
        
        # Apply mask (only compute loss on positive samples)
        theta_pred = theta_pred[mask > 0]
        r_pred = r_pred[mask > 0]
        theta_target = theta_target[mask > 0]
        r_target = r_target[mask > 0]
        
        if theta_pred.numel() == 0:
            return torch.tensor(0.0, device=theta_pred.device)
        
        # Smooth L1 loss for theta
        theta_loss = F.smooth_l1_loss(theta_pred, theta_target, beta=self.beta, reduction=self.reduction)
        
        # Smooth L1 loss for r
        r_loss = F.smooth_l1_loss(r_pred, r_target, beta=self.beta, reduction=self.reduction)
        
        # Combined loss
        total_loss = theta_loss + r_loss
        
        return total_loss


class AFPLLoss(nn.Module):
    """
    Overall loss for AFPL-Net
    
    Combines three loss components:
    1. Focal Loss for classification
    2. BCE Loss for centerness
    3. Smooth L1 Loss for polar regression
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Loss weights
        self.cls_weight = cfg.cls_loss_weight if hasattr(cfg, 'cls_loss_weight') else 1.0
        self.centerness_weight = cfg.centerness_loss_weight if hasattr(cfg, 'centerness_loss_weight') else 1.0
        self.regression_weight = cfg.regression_loss_weight if hasattr(cfg, 'regression_loss_weight') else 1.0
        
        # Loss functions
        self.focal_loss = FocalLoss(
            alpha=cfg.cls_loss_alpha if hasattr(cfg, 'cls_loss_alpha') else 0.25,
            gamma=cfg.cls_loss_gamma if hasattr(cfg, 'cls_loss_gamma') else 2.0
        )
        self.centerness_loss = CenternessLoss()
        self.polar_regression_loss = PolarRegressionLoss(
            beta=cfg.regression_beta if hasattr(cfg, 'regression_beta') else 1.0
        )
        
    def forward(self, pred_dict, target_dict):
        """
        Compute overall loss
        
        Args:
            pred_dict: Dictionary with predictions
                - cls_pred: [B, 1, H, W]
                - centerness_pred: [B, 1, H, W]
                - theta_pred: [B, 1, H, W]
                - r_pred: [B, 1, H, W]
            target_dict: Dictionary with ground truth
                - cls_gt: [B, H, W]
                - centerness_gt: [B, H, W]
                - theta_gt: [B, H, W]
                - r_gt: [B, H, W]
                
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss values
        """
        # Extract predictions
        cls_pred = pred_dict['cls_pred']
        centerness_pred = pred_dict['centerness_pred']
        theta_pred = pred_dict['theta_pred']
        r_pred = pred_dict['r_pred']
        
        # Extract ground truth
        cls_gt = target_dict['cls_gt']
        centerness_gt = target_dict['centerness_gt']
        theta_gt = target_dict['theta_gt']
        r_gt = target_dict['r_gt']
        
        # Compute individual losses
        loss_cls = self.focal_loss(cls_pred, cls_gt)
        loss_centerness = self.centerness_loss(centerness_pred, centerness_gt, mask=cls_gt)
        loss_regression = self.polar_regression_loss(theta_pred, r_pred, theta_gt, r_gt, mask=cls_gt)
        
        # Combine losses
        total_loss = (
            self.cls_weight * loss_cls +
            self.centerness_weight * loss_centerness +
            self.regression_weight * loss_regression
        )
        
        # Create loss dictionary for logging
        loss_dict = {
            'loss_cls': loss_cls.item(),
            'loss_centerness': loss_centerness.item(),
            'loss_regression': loss_regression.item(),
            'loss': total_loss.item()
        }
        
        return total_loss, loss_dict
