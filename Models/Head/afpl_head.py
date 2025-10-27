"""
AFPL Head: Anchor-Free Polar Lane Detection Head

This head implements three parallel prediction branches for single-stage,
anchor-free lane detection using polar coordinates:
1. Classification Head: Predicts if a pixel belongs to any lane line
2. Centerness Head: Predicts the quality of the lane point
3. Polar Regression Head: Predicts (θ, r) polar coordinates relative to global pole
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import numpy as np


class AFPLHead(nn.Module):
    """
    Anchor-Free Polar Lane Detection Head
    
    Combines ideas from PolarMask and Polar R-CNN:
    - Uses global pole (vanishing point) like Polar R-CNN
    - Uses per-pixel prediction like PolarMask (anchor-free, single-stage)
    - Uses centerness to improve prediction quality
    """
    
    def __init__(self, cfg):
        super(AFPLHead, self).__init__()
        
        # Image and network parameters
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.in_channels = cfg.neck_dim  # FPN output channels
        
        # Global pole (vanishing point) - fixed reference point
        self.global_pole_x = cfg.center_w if hasattr(cfg, 'center_w') else cfg.img_w // 2
        self.global_pole_y = cfg.center_h if hasattr(cfg, 'center_h') else cfg.img_h // 4
        
        # Inference parameters
        self.conf_threshold = cfg.conf_thres if hasattr(cfg, 'conf_thres') else 0.1
        self.centerness_threshold = cfg.centerness_thres if hasattr(cfg, 'centerness_thres') else 0.1
        self.angle_cluster_eps = cfg.angle_cluster_eps if hasattr(cfg, 'angle_cluster_eps') else 0.035  # ~2 degrees
        self.min_cluster_points = cfg.min_cluster_points if hasattr(cfg, 'min_cluster_points') else 10
        
        # Build three parallel prediction heads
        self._build_classification_head()
        self._build_centerness_head()
        self._build_polar_regression_head()
        
    def _build_classification_head(self):
        """
        Classification Head: H×W×1
        Predicts whether each pixel belongs to any lane line
        Uses Focal Loss during training
        """
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // 2, 1, 1, 1, 0)
        )
        
    def _build_centerness_head(self):
        """
        Centerness Head: H×W×1
        Predicts the quality of lane point (higher if closer to lane centerline)
        Uses BCE Loss during training
        """
        self.centerness_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // 2, 1, 1, 1, 0)
        )
        
    def _build_polar_regression_head(self):
        """
        Polar Regression Head: H×W×2
        Predicts (θ, r) polar coordinates of each pixel relative to global pole
        Uses Smooth L1 Loss during training
        """
        self.polar_reg_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // 2, 2, 1, 1, 0)  # Output: (θ, r)
        )
        
    def forward(self, feats):
        """
        Forward pass through AFPL head
        
        Args:
            feats: List of FPN feature maps [P3, P4, P5, ...]
            
        Returns:
            Dictionary containing predictions from all three heads
        """
        # Use the highest resolution feature map (P3)
        # For multi-scale, we could process all levels
        feat = feats[0]  # Assuming P3 is first
        
        # Three parallel predictions
        cls_pred = self.cls_head(feat)  # [B, 1, H, W]
        centerness_pred = self.centerness_head(feat)  # [B, 1, H, W]
        polar_pred = self.polar_reg_head(feat)  # [B, 2, H, W]
        
        # Normalize polar predictions
        # θ: normalize to [-π, π]
        # r: keep as raw output, will be interpreted as distance in pixels
        theta_pred = torch.tanh(polar_pred[:, 0:1, ...]) * math.pi  # [-π, π]
        r_pred = F.relu(polar_pred[:, 1:2, ...])  # [0, ∞)
        
        pred_dict = {
            'cls_pred': cls_pred,  # [B, 1, H, W]
            'centerness_pred': centerness_pred,  # [B, 1, H, W]
            'theta_pred': theta_pred,  # [B, 1, H, W]
            'r_pred': r_pred,  # [B, 1, H, W]
        }
        
        return pred_dict
    
    def compute_ground_truth(self, lane_masks, center_distances=None):
        """
        Compute ground truth for training
        
        Args:
            lane_masks: Binary masks for lane lines [B, H, W] or [B, N, H, W]
            center_distances: Distance to lane centerline [B, H, W] (optional)
            
        Returns:
            Dictionary containing GT for all three heads
        """
        device = lane_masks.device
        B, H, W = lane_masks.shape[:3] if lane_masks.dim() == 3 else (lane_masks.shape[0], lane_masks.shape[2], lane_masks.shape[3])
        
        # Classification GT: 1 if pixel is on any lane, 0 otherwise
        if lane_masks.dim() == 4:  # [B, N, H, W] - multiple lanes
            cls_gt = (lane_masks.sum(dim=1) > 0).float()  # [B, H, W]
        else:
            cls_gt = lane_masks.float()  # [B, H, W]
        
        # Centerness GT: Gaussian based on distance to centerline
        if center_distances is not None:
            sigma = 10.0  # Tunable parameter
            centerness_gt = torch.exp(-center_distances ** 2 / (2 * sigma ** 2))
        else:
            # If no distance info, use binary mask as approximation
            centerness_gt = cls_gt
        
        # Polar coordinate GT: compute (θ, r) for positive pixels
        theta_gt = torch.zeros((B, H, W), device=device)
        r_gt = torch.zeros((B, H, W), device=device)
        
        # Create pixel coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Compute polar coordinates relative to global pole
        dx = x_coords - self.global_pole_x
        dy = y_coords - self.global_pole_y
        
        theta = torch.atan2(dy, dx)  # [-π, π]
        r = torch.sqrt(dx ** 2 + dy ** 2)
        
        # Only valid for positive samples
        theta_gt = theta.unsqueeze(0).expand(B, -1, -1)
        r_gt = r.unsqueeze(0).expand(B, -1, -1)
        
        gt_dict = {
            'cls_gt': cls_gt,  # [B, H, W]
            'centerness_gt': centerness_gt,  # [B, H, W]
            'theta_gt': theta_gt,  # [B, H, W]
            'r_gt': r_gt,  # [B, H, W]
        }
        
        return gt_dict
    
    def post_process(self, pred_dict, downsample_factor=8):
        """
        Post-processing: Angular clustering and lane formation
        
        This is the key innovation - no NMS needed!
        Points with similar θ automatically belong to the same lane.
        
        Args:
            pred_dict: Predictions from forward pass
            downsample_factor: Feature map downsample factor relative to input
            
        Returns:
            List of lane curves (one per batch item)
        """
        cls_pred = torch.sigmoid(pred_dict['cls_pred'])
        centerness_pred = torch.sigmoid(pred_dict['centerness_pred'])
        theta_pred = pred_dict['theta_pred']
        r_pred = pred_dict['r_pred']
        
        batch_size = cls_pred.shape[0]
        lanes_batch = []
        
        for b in range(batch_size):
            # Compute final score: classification × centerness
            final_score = cls_pred[b, 0] * centerness_pred[b, 0]  # [H, W]
            
            # Filter low-confidence points
            valid_mask = final_score > self.conf_threshold
            
            if not valid_mask.any():
                lanes_batch.append([])
                continue
            
            # Get coordinates of valid points
            y_coords, x_coords = torch.where(valid_mask)
            
            # Get predictions for valid points
            scores = final_score[valid_mask].cpu().numpy()
            thetas = theta_pred[b, 0, valid_mask].cpu().numpy()
            rs = r_pred[b, 0, valid_mask].cpu().numpy()
            
            # Convert feature map coordinates to image coordinates
            x_img = x_coords.cpu().numpy() * downsample_factor
            y_img = y_coords.cpu().numpy() * downsample_factor
            
            # Angular clustering: group points with similar θ
            lanes = self._cluster_by_angle(thetas, rs, x_img, y_img, scores)
            lanes_batch.append(lanes)
        
        return lanes_batch
    
    def _cluster_by_angle(self, thetas, rs, x_coords, y_coords, scores):
        """
        Cluster points by angle using DBSCAN
        
        Points with similar θ belong to the same lane.
        """
        if len(thetas) == 0:
            return []
        
        # Use θ as primary clustering feature
        # Normalize angle to be between 0 and 2π for clustering
        thetas_normalized = (thetas + math.pi) / (2 * math.pi)  # [0, 1]
        
        # Reshape for DBSCAN
        features = thetas_normalized.reshape(-1, 1)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=self.angle_cluster_eps, min_samples=self.min_cluster_points)
        labels = clustering.fit_predict(features)
        
        # Form lanes from clusters
        lanes = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in unique_labels:
            cluster_mask = labels == label
            
            # Get points in this cluster
            cluster_thetas = thetas[cluster_mask]
            cluster_rs = rs[cluster_mask]
            cluster_x = x_coords[cluster_mask]
            cluster_y = y_coords[cluster_mask]
            cluster_scores = scores[cluster_mask]
            
            # Sort by radius (distance from pole)
            sort_idx = np.argsort(cluster_rs)
            cluster_thetas = cluster_thetas[sort_idx]
            cluster_rs = cluster_rs[sort_idx]
            cluster_x = cluster_x[sort_idx]
            cluster_y = cluster_y[sort_idx]
            cluster_scores = cluster_scores[sort_idx]
            
            # Convert polar to Cartesian if needed
            # (Already have x, y from feature map, but could recompute for consistency)
            lane_x = cluster_rs * np.cos(cluster_thetas) + self.global_pole_x
            lane_y = cluster_rs * np.sin(cluster_thetas) + self.global_pole_y
            
            # Create lane curve as list of (x, y) points
            lane_points = list(zip(lane_x, lane_y))
            lanes.append({
                'points': lane_points,
                'scores': cluster_scores,
                'mean_score': np.mean(cluster_scores)
            })
        
        # Sort lanes by mean score (descending)
        lanes.sort(key=lambda x: x['mean_score'], reverse=True)
        
        return lanes
