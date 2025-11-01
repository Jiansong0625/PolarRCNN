"""
Loss functions for AFPL-Net

Implements enhanced loss components inspired by polar-based methods:
1. Quality Focal Loss for classification (inspired by FCOS/PolarMask)
2. BCE Loss for centerness (point quality)
3. Polar IoU Loss for regression (inspired by PolarMask)
4. Smooth L1 Loss for polar regression (θ, r) with uncertainty weighting
5. Gradient normalization for balanced training

Key improvements over baseline:
- Polar IoU loss: Better geometric understanding of (θ, r) predictions
- Quality weighting: Focus on high-quality predictions
- Uncertainty weighting: Automatic balancing between theta and radius losses
- Gradient normalization: Prevents any single loss from dominating
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


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


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss - integrates object quality (IoU/centerness) into focal loss
    
    Inspired by:
    - FCOS: "FCOS: Fully Convolutional One-Stage Object Detection"
    - PolarMask: quality-aware classification
    
    The key idea: train classification to predict quality score (e.g., centerness)
    instead of just binary presence. This produces better quality estimates and
    suppresses low-quality predictions.
    """
    
    def __init__(self, beta=2.0, reduction='mean'):
        super().__init__()
        self.beta = beta  # Focusing parameter (similar to gamma in focal loss)
        self.reduction = reduction
    
    def forward(self, pred, target, quality_target=None):
        """
        Args:
            pred: Predicted logits [B, 1, H, W] or [B, H, W]
            target: Binary ground truth [B, H, W] (0 or 1)
            quality_target: Quality score [B, H, W] (e.g., centerness, 0-1)
                           If None, falls back to binary target
            
        Returns:
            Quality focal loss value
        """
        # Flatten tensors
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        if quality_target is None:
            quality_target = target
        else:
            quality_target = quality_target.reshape(-1)
        
        # For negative samples, quality = 0; for positive samples, quality = centerness/IoU
        quality_target = quality_target * target
        
        pred_prob = torch.sigmoid(pred)
        
        # Quality focal loss formulation:
        # For negatives (target=0): standard focal loss
        # For positives (target=1): weight by |pred - quality|^beta
        
        # Cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, quality_target, reduction='none')
        
        # Modulating factor
        # For negatives: (1 - pred_prob)^beta focuses on hard negatives
        # For positives: |pred_prob - quality|^beta focuses on quality mismatch
        modulation = torch.where(
            target > 0,
            torch.abs(pred_prob - quality_target) ** self.beta,
            (1 - pred_prob) ** self.beta
        )
        
        loss = modulation * ce_loss
        
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
    Smooth L1 Loss for polar coordinate regression with uncertainty weighting
    
    Predicts (θ, r) for each lane point relative to global pole.
    Only computed for positive samples (lane pixels).
    
    Enhanced with uncertainty-based weighting to automatically balance
    theta and radius losses based on prediction confidence.
    """
    
    def __init__(self, beta=1.0, use_uncertainty=False, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.use_uncertainty = use_uncertainty
        self.reduction = reduction
        
        # Learnable uncertainty parameters (log variance)
        # Using log variance for numerical stability
        if use_uncertainty:
            self.log_var_theta = nn.Parameter(torch.zeros(1))
            self.log_var_r = nn.Parameter(torch.zeros(1))
        
    def forward(self, theta_pred, r_pred, theta_target, r_target, mask):
        """
        Args:
            theta_pred: Predicted angle [B, 1, H, W]
            r_pred: Predicted radius [B, 1, H, W]
            theta_target: Target angle [B, H, W]
            r_target: Target radius [B, H, W]
            mask: Positive sample mask [B, H, W]
            
        Returns:
            Smooth L1 loss value (uncertainty-weighted if enabled)
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
        
        if self.use_uncertainty:
            # Uncertainty weighting: Loss = 1/(2*sigma^2) * loss + log(sigma)
            # Using log_var = log(sigma^2) for stability
            # This automatically balances theta vs radius based on task difficulty
            precision_theta = torch.exp(-self.log_var_theta)
            precision_r = torch.exp(-self.log_var_r)
            
            total_loss = (
                precision_theta * theta_loss + self.log_var_theta +
                precision_r * r_loss + self.log_var_r
            )
        else:
            # Standard equal weighting
            total_loss = theta_loss + r_loss
        
        return total_loss


class PolarIoULoss(nn.Module):
    """
    Polar IoU Loss for better geometric understanding
    
    Inspired by PolarMask's polar IoU computation. This loss directly
    optimizes the overlap between predicted and ground truth polar regions,
    providing better geometric guidance than pure coordinate regression.
    
    The key insight: (θ, r) coordinates define a polar region. We want to
    maximize overlap between predicted and ground truth regions.
    """
    
    def __init__(self, loss_type='iou', eps=1e-6, reduction='mean'):
        super().__init__()
        self.loss_type = loss_type  # 'iou' or 'giou'
        self.eps = eps
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
            Polar IoU loss (1 - IoU or 1 - GIoU)
        """
        # Flatten tensors
        theta_pred = theta_pred.squeeze(1).reshape(-1)
        r_pred = r_pred.squeeze(1).reshape(-1)
        theta_target = theta_target.reshape(-1)
        r_target = r_target.reshape(-1)
        mask = mask.reshape(-1)
        
        # Apply mask
        theta_pred = theta_pred[mask > 0]
        r_pred = r_pred[mask > 0]
        theta_target = theta_target[mask > 0]
        r_target = r_target[mask > 0]
        
        if theta_pred.numel() == 0:
            return torch.tensor(0.0, device=theta_pred.device)
        
        # Clamp predictions to valid ranges
        r_pred = torch.clamp(r_pred, min=self.eps)
        r_target = torch.clamp(r_target, min=self.eps)
        
        # Compute polar IoU
        # For each point, the "area" is approximated by r^2 * δθ
        # Where δθ is a small angular width around the point
        
        # Compute intersection: min of predicted and target radii
        # The angular overlap is computed based on angle difference
        r_min = torch.min(r_pred, r_target)
        r_max = torch.max(r_pred, r_target)
        
        # Angular distance (wrapped to [-π, π])
        angle_diff = theta_pred - theta_target
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Approximate area overlap based on radial and angular components
        # For small angles, area ≈ r^2 * θ / 2 (sector area)
        # Intersection area uses min radius
        # Union area uses max radius
        
        # Radial IoU component (r intersection over union)
        r_iou = r_min / (r_max + self.eps)
        
        # Angular component: penalize angle differences
        # Use a smooth decay function: exp(-|angle_diff|)
        # This makes points with similar angles have higher overlap
        angle_weight = torch.exp(-torch.abs(angle_diff))
        
        # Combined Polar IoU
        polar_iou = r_iou * angle_weight
        
        if self.loss_type == 'giou':
            # Generalized IoU: add penalty for distance in polar space
            # Enclosing "box" in polar space
            r_enclosing = r_max
            
            # GIoU penalty term
            c_area = r_enclosing * r_enclosing  # Enclosing area
            union_area = r_max * r_max
            giou_penalty = (c_area - union_area) / (c_area + self.eps)
            
            polar_iou = polar_iou - giou_penalty
        
        # IoU loss: 1 - IoU
        loss = 1.0 - polar_iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AFPLLoss(nn.Module):
    """
    Enhanced loss for AFPL-Net with multiple improvements
    
    Combines multiple loss components:
    1. Quality Focal Loss / Focal Loss for classification
    2. BCE Loss for centerness
    3. Polar IoU Loss for geometric understanding (optional)
    4. Smooth L1 Loss for polar regression
    5. Gradient normalization for balanced training
    
    Key improvements:
    - Quality-aware classification (integrates centerness into classification)
    - Polar IoU for better geometric optimization
    - Uncertainty weighting for automatic theta/r balancing
    - Gradient normalization to prevent loss domination
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Loss weights
        self.cls_weight = cfg.cls_loss_weight if hasattr(cfg, 'cls_loss_weight') else 1.0
        self.centerness_weight = cfg.centerness_loss_weight if hasattr(cfg, 'centerness_loss_weight') else 1.0
        self.regression_weight = cfg.regression_loss_weight if hasattr(cfg, 'regression_loss_weight') else 1.0
        
        # Additional weights for new loss components
        self.polar_iou_weight = cfg.polar_iou_weight if hasattr(cfg, 'polar_iou_weight') else 0.5
        
        # Loss function selection
        self.use_quality_focal = cfg.use_quality_focal if hasattr(cfg, 'use_quality_focal') else False
        self.use_polar_iou = cfg.use_polar_iou if hasattr(cfg, 'use_polar_iou') else True
        self.use_uncertainty = cfg.use_uncertainty if hasattr(cfg, 'use_uncertainty') else False
        self.use_grad_norm = cfg.use_grad_norm if hasattr(cfg, 'use_grad_norm') else True
        
        # Classification loss
        if self.use_quality_focal:
            self.focal_loss = QualityFocalLoss(
                beta=cfg.cls_loss_gamma if hasattr(cfg, 'cls_loss_gamma') else 2.0
            )
        else:
            self.focal_loss = FocalLoss(
                alpha=cfg.cls_loss_alpha if hasattr(cfg, 'cls_loss_alpha') else 0.25,
                gamma=cfg.cls_loss_gamma if hasattr(cfg, 'cls_loss_gamma') else 2.0
            )
        
        # Centerness loss
        self.centerness_loss = CenternessLoss()
        
        # Regression losses
        self.polar_regression_loss = PolarRegressionLoss(
            beta=cfg.regression_beta if hasattr(cfg, 'regression_beta') else 1.0,
            use_uncertainty=self.use_uncertainty
        )
        
        if self.use_polar_iou:
            self.polar_iou_loss = PolarIoULoss(
                loss_type=cfg.polar_iou_type if hasattr(cfg, 'polar_iou_type') else 'iou'
            )
        
        # For gradient normalization - store running means
        if self.use_grad_norm:
            self.register_buffer('grad_norm_alpha', torch.tensor(0.9))
            self.register_buffer('loss_cls_mean', torch.tensor(1.0))
            self.register_buffer('loss_centerness_mean', torch.tensor(1.0))
            self.register_buffer('loss_regression_mean', torch.tensor(1.0))
            if self.use_polar_iou:
                self.register_buffer('loss_polar_iou_mean', torch.tensor(1.0))
        
    def gradient_normalize(self, losses_dict):
        """
        Gradient normalization to balance loss components
        
        Inspired by GradNorm and used in many modern detectors.
        Normalizes losses by their running mean to prevent any single
        loss from dominating the gradient flow.
        """
        if not self.training or not self.use_grad_norm:
            return losses_dict
        
        # Update running means with exponential moving average
        with torch.no_grad():
            self.loss_cls_mean = (
                self.grad_norm_alpha * self.loss_cls_mean + 
                (1 - self.grad_norm_alpha) * losses_dict['loss_cls'].detach()
            )
            self.loss_centerness_mean = (
                self.grad_norm_alpha * self.loss_centerness_mean + 
                (1 - self.grad_norm_alpha) * losses_dict['loss_centerness'].detach()
            )
            self.loss_regression_mean = (
                self.grad_norm_alpha * self.loss_regression_mean + 
                (1 - self.grad_norm_alpha) * losses_dict['loss_regression'].detach()
            )
            if self.use_polar_iou and 'loss_polar_iou' in losses_dict:
                self.loss_polar_iou_mean = (
                    self.grad_norm_alpha * self.loss_polar_iou_mean + 
                    (1 - self.grad_norm_alpha) * losses_dict['loss_polar_iou'].detach()
                )
        
        # Normalize losses by their running means
        normalized_losses = {}
        normalized_losses['loss_cls'] = losses_dict['loss_cls'] / (self.loss_cls_mean + 1e-6)
        normalized_losses['loss_centerness'] = losses_dict['loss_centerness'] / (self.loss_centerness_mean + 1e-6)
        normalized_losses['loss_regression'] = losses_dict['loss_regression'] / (self.loss_regression_mean + 1e-6)
        
        if self.use_polar_iou and 'loss_polar_iou' in losses_dict:
            normalized_losses['loss_polar_iou'] = losses_dict['loss_polar_iou'] / (self.loss_polar_iou_mean + 1e-6)
        
        return normalized_losses
        
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
        if self.use_quality_focal:
            # Quality focal loss: use centerness as quality target
            loss_cls = self.focal_loss(cls_pred, cls_gt, quality_target=centerness_gt)
        else:
            loss_cls = self.focal_loss(cls_pred, cls_gt)
        
        loss_centerness = self.centerness_loss(centerness_pred, centerness_gt, mask=cls_gt)
        loss_regression = self.polar_regression_loss(theta_pred, r_pred, theta_gt, r_gt, mask=cls_gt)
        
        # Store raw losses
        raw_losses = {
            'loss_cls': loss_cls,
            'loss_centerness': loss_centerness,
            'loss_regression': loss_regression
        }
        
        # Optionally add polar IoU loss
        if self.use_polar_iou:
            loss_polar_iou = self.polar_iou_loss(theta_pred, r_pred, theta_gt, r_gt, mask=cls_gt)
            raw_losses['loss_polar_iou'] = loss_polar_iou
        
        # Apply gradient normalization if enabled
        if self.use_grad_norm and self.training:
            normalized_losses = self.gradient_normalize(raw_losses)
            loss_cls_weighted = self.cls_weight * normalized_losses['loss_cls']
            loss_centerness_weighted = self.centerness_weight * normalized_losses['loss_centerness']
            loss_regression_weighted = self.regression_weight * normalized_losses['loss_regression']
            
            total_loss = loss_cls_weighted + loss_centerness_weighted + loss_regression_weighted
            
            if self.use_polar_iou:
                loss_polar_iou_weighted = self.polar_iou_weight * normalized_losses['loss_polar_iou']
                total_loss = total_loss + loss_polar_iou_weighted
        else:
            # Standard weighted combination
            total_loss = (
                self.cls_weight * loss_cls +
                self.centerness_weight * loss_centerness +
                self.regression_weight * loss_regression
            )
            
            if self.use_polar_iou:
                total_loss = total_loss + self.polar_iou_weight * raw_losses['loss_polar_iou']
        
        # Create loss dictionary for logging (use raw loss values)
        loss_dict = {
            'loss_cls': loss_cls.item() if torch.is_tensor(loss_cls) else loss_cls,
            'loss_centerness': loss_centerness.item() if torch.is_tensor(loss_centerness) else loss_centerness,
            'loss_regression': loss_regression.item() if torch.is_tensor(loss_regression) else loss_regression,
            'loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss
        }
        
        if self.use_polar_iou:
            loss_dict['loss_polar_iou'] = raw_losses['loss_polar_iou'].item() if torch.is_tensor(raw_losses['loss_polar_iou']) else raw_losses['loss_polar_iou']
        
        # Add uncertainty parameters to loss dict if using uncertainty weighting
        if self.use_uncertainty and hasattr(self.polar_regression_loss, 'log_var_theta'):
            loss_dict['sigma_theta'] = torch.exp(0.5 * self.polar_regression_loss.log_var_theta).item()
            loss_dict['sigma_r'] = torch.exp(0.5 * self.polar_regression_loss.log_var_r).item()
        
        return total_loss, loss_dict
