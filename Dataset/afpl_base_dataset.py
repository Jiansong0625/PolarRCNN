"""
AFPL-Net specific dataset base classes

These datasets generate ground truth suitable for AFPL-Net (single-stage, anchor-free),
which is different from the two-stage Polar R-CNN ground truth.

AFPL-Net needs:
- cls_gt: Binary lane mask [B, H, W]
- centerness_gt: Centerness values [B, H, W]
- theta_gt: Polar angles [B, H, W]
- r_gt: Polar radii [B, H, W]
"""

import torch
from torch.utils.data import Dataset
import math
import random
import numpy as np
import albumentations as A
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from utils.lane_utils import clipline_out_of_image
import cv2


class AFPLBaseTrSet(Dataset):
    """Base training dataset for AFPL-Net"""
    
    def __init__(self, cfg=None, transforms=None):
        self.cfg = cfg
        random.seed(cfg.random_seed)
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.center_h, self.center_w = cfg.center_h, cfg.center_w
        self.max_lanes = cfg.max_lanes
        self.transforms = transforms
        
        # Feature map downsample factor
        self.downsample_factor = cfg.downsample_strides[0] if hasattr(cfg, 'downsample_strides') else 8
        self.feat_h = self.img_h // self.downsample_factor
        self.feat_w = self.img_w // self.downsample_factor
        
        # Setup data augmentation
        img_transforms = []
        self.aug_names = cfg.train_augments
        for aug in self.aug_names:
            if aug['name'] != 'OneOf':
                img_transforms.append(getattr(A, aug['name'])(**aug['parameters']))
            else:
                img_transforms.append(A.OneOf([getattr(A, aug_['name'])(**aug_['parameters'])
                                      for aug_ in aug['transforms']], p=aug['p']))
        self.train_augments = A.Compose(img_transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img, lanes = self.get_sample(index)
        img, lanes, transformed_center = self.augment(img, lanes)
        
        # Convert image to tensor
        data_dict = dict()
        data_dict['img'] = self.transforms(img)
        
        # Generate AFPL-Net ground truth using the transformed center point
        cls_gt, centerness_gt, theta_gt, r_gt = self.generate_afpl_ground_truth(lanes, transformed_center)
        
        data_dict['cls_gt'] = cls_gt
        data_dict['centerness_gt'] = centerness_gt
        data_dict['theta_gt'] = theta_gt
        data_dict['r_gt'] = r_gt
        
        return data_dict
    
    def generate_afpl_ground_truth(self, lanes, transformed_center):
        """
        Generate ground truth for AFPL-Net from lane annotations
        
        Args:
            lanes: List of lane arrays, each with shape [N, 2] (x, y coordinates)
            transformed_center: The center point after augmentation [x, y]
            
        Returns:
            cls_gt: Binary lane mask [feat_h, feat_w]
            centerness_gt: Centerness values [feat_h, feat_w]
            theta_gt: Polar angles [feat_h, feat_w]
            r_gt: Polar radii [feat_h, feat_w]
        """
        # Initialize ground truth arrays at feature map resolution
        cls_gt = np.zeros((self.feat_h, self.feat_w), dtype=np.float32)
        centerness_gt = np.zeros((self.feat_h, self.feat_w), dtype=np.float32)
        theta_gt = np.zeros((self.feat_h, self.feat_w), dtype=np.float32)
        r_gt = np.zeros((self.feat_h, self.feat_w), dtype=np.float32)
        
        # Precompute polar coordinates for all pixels at feature map resolution
        # Note: Use transformed center coordinates (already in image space), scale to feature map space
        center_w_feat = transformed_center[0] / self.downsample_factor
        center_h_feat = transformed_center[1] / self.downsample_factor
        
        y_coords, x_coords = np.meshgrid(
            np.arange(self.feat_h, dtype=np.float32),
            np.arange(self.feat_w, dtype=np.float32),
            indexing='ij'
        )
        
        dx = x_coords - center_w_feat
        dy = y_coords - center_h_feat
        theta = np.arctan2(dy, dx)  # [-π, π]
        r = np.sqrt(dx ** 2 + dy ** 2) * self.downsample_factor  # Convert to image space distance
        
        # Store precomputed values
        theta_gt = theta
        r_gt = r
        
        # For each lane, create a mask and centerness at feature map resolution
        for lane in lanes:
            if len(lane) < 2:
                continue
            
            # Scale lane coordinates to feature map resolution
            lane_feat = lane / self.downsample_factor
            
            # Draw lane as a thick line at feature map resolution
            lane_mask = np.zeros((self.feat_h, self.feat_w), dtype=np.uint8)
            lane_int = lane_feat.astype(np.int32)
            
            # Draw lines between consecutive points (thickness adjusted for feature map scale)
            for i in range(len(lane_int) - 1):
                pt1 = tuple(lane_int[i])
                pt2 = tuple(lane_int[i + 1])
                # Use thickness ~2 pixels at feature map resolution (equivalent to ~16 at image resolution)
                cv2.line(lane_mask, pt1, pt2, 1, thickness=2)
            
            # Update classification ground truth
            cls_gt = np.maximum(cls_gt, lane_mask.astype(np.float32))
            
            # Compute centerness: distance to lane centerline
            lane_distance = cv2.distanceTransform(
                (1 - lane_mask).astype(np.uint8), 
                cv2.DIST_L2, 
                cv2.DIST_MASK_PRECISE
            )
            
            # Convert distance to centerness (Gaussian-like)
            # Sigma adjusted for feature map scale
            sigma = 1.0  # Tunable parameter (at feature map scale)
            lane_centerness = np.exp(-lane_distance ** 2 / (2 * sigma ** 2))
            lane_centerness = lane_centerness * lane_mask  # Only on lane pixels
            
            # Take maximum centerness across lanes
            centerness_gt = np.maximum(centerness_gt, lane_centerness)
        
        return cls_gt, centerness_gt, theta_gt, r_gt
    
    def augment(self, img, lanes):
        """Apply data augmentation"""
        # Add center point as a keypoint to track its transformation
        center_point = np.array([[self.center_w, self.center_h]], dtype=np.float32)
        
        if len(lanes) > 0:
            lane_lengths = [len(lane) for lane in lanes]
            keypoints = np.concatenate(lanes, axis=0)
            # Append center point to keypoints
            keypoints = np.concatenate([keypoints, center_point], axis=0)
            content = self.train_augments(image=img, keypoints=keypoints)
            keypoints = np.array(content['keypoints'])
            
            # Extract transformed center point (it's the last keypoint)
            transformed_center = keypoints[-1]
            # Remove center point from keypoints
            keypoints = keypoints[:-1]
            
            start_dim = 0
            lanes = []
            for lane_length in lane_lengths:
                lane = keypoints[start_dim:start_dim+lane_length]
                lanes.append(lane)
                start_dim += lane_length
        else:
            # Even with no lanes, we need to track center point transformation
            content = self.train_augments(image=img, keypoints=center_point)
            transformed_center = np.array(content['keypoints'])[0]
        
        img = content['image']
        
        # Clip lanes to image boundaries
        clip_lanes = []
        img_shape = (img.shape[0], img.shape[1])
        for lane in lanes:
            lane = clipline_out_of_image(line_coords=lane, img_shape=img_shape)
            if lane is not None and len(lane) > 1:
                clip_lanes.append(lane)
        lanes = clip_lanes
        
        # Return augmented image, lanes, and transformed center point
        return img, lanes, transformed_center
    
    def collate_fn(self, data_dict_list):
        """Collate batch of samples"""
        batch_dict = {}
        
        # Stack all tensors
        for key in data_dict_list[0].keys():
            if key == 'img':
                # Images are already tensors from transforms
                batch_dict[key] = torch.stack([d[key] for d in data_dict_list], dim=0)
            else:
                # Convert numpy arrays to tensors
                batch_dict[key] = torch.stack([
                    torch.from_numpy(d[key]) for d in data_dict_list
                ], dim=0)
        
        return batch_dict


class AFPLBaseTsSet(Dataset):
    """Base test dataset for AFPL-Net (same as original, no GT needed)"""
    
    def __init__(self, cfg=None, transforms=None):
        self.data_root = cfg.data_root
        self.cut_height = cfg.cut_height
        self.transforms = transforms
        self.cut_height = cfg.cut_height
        self.is_val = cfg.is_val
        self.is_view = cfg.is_view
        self.img_path_list = []
        self.file_name_list = []

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        ori_img = cv2.imread(self.img_path_list[index])
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img = ori_img[self.cut_height:]
        if self.transforms is not None:
            img = self.transforms(img)
        if not self.is_view:
            ori_img = None
        return img, self.file_name_list[index], ori_img

    def collate_fn(self, samples):
        img_list = []
        ori_img_list = []
        file_name_list = []
        for img, file_name, ori_img in samples:
            img_list.append(img.unsqueeze(0))
            file_name_list.append(file_name)
            if ori_img is not None:
                ori_img_list.append(torch.from_numpy(ori_img))
        imgs = torch.cat(img_list, dim=0)
        return imgs, file_name_list, ori_img_list
