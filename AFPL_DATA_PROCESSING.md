# AFPL-Net Data Processing Implementation

## Overview

This document describes the implementation of AFPL-Net specific data processing that enables training and testing with the single-stage, anchor-free AFPL-Net architecture.

## Problem

Previously, AFPL-Net was using data processing code designed for the two-stage Polar R-CNN detector, which:
- Required `num_line_groups` and `polar_map_size` parameters (not needed for AFPL-Net)
- Generated ground truth like `line_paras`, `line_paras_group`, `polar_map` (two-stage specific)
- AFPL-Net actually needs: `cls_gt`, `centerness_gt`, `theta_gt`, `r_gt`

## Solution

### 1. New AFPL-Specific Base Dataset Classes

**File:** `Dataset/afpl_base_dataset.py`

Created `AFPLBaseTrSet` that:
- Does NOT require two-stage parameters (`num_line_groups`, `polar_map_size`)
- Generates ground truth at feature map resolution (H/8 × W/8 for downsample_stride=8)
- Produces the correct ground truth format:
  - `cls_gt`: Binary lane mask [feat_h, feat_w]
  - `centerness_gt`: Centerness values [feat_h, feat_w]
  - `theta_gt`: Polar angles [-π, π] [feat_h, feat_w]
  - `r_gt`: Polar radii [feat_h, feat_w]

### 2. CULane Dataset Implementation

**File:** `Dataset/afpl_culane_dataset.py`

Created AFPL-specific CULane dataset classes:
- `AFPLCULaneTrSet`: Training dataset
- `AFPLCULaneTsSet`: Testing dataset

These inherit from the AFPL base classes and implement CULane-specific data loading.

### 3. Updated Dataset Builder

**File:** `Dataset/build.py`

Modified `build_trainset()` and `build_testset()` to:
- Detect if config is for AFPL-Net (checks `cfg_name` contains 'afplnet')
- Use AFPL-specific datasets when appropriate
- Fall back to original Polar R-CNN datasets otherwise

```python
# Check if using AFPL-Net
is_afpl = hasattr(cfg, 'cfg_name') and 'afplnet' in cfg.cfg_name.lower()

if cfg.dataset == 'culane':
    if is_afpl:
        from .afpl_culane_dataset import AFPLCULaneTrSet
        trainset = AFPLCULaneTrSet(cfg=cfg, transforms=transform)
    else:
        from .culane_dataset import CULaneTrSet
        trainset = CULaneTrSet(cfg=cfg, transforms=transform)
```

## Key Implementation Details

### Ground Truth Resolution

AFPL-Net predictions are at feature map resolution (downsampled 8x), so ground truth must match:

```python
# Feature map dimensions
self.downsample_factor = cfg.downsample_strides[0]  # 8
self.feat_h = self.img_h // self.downsample_factor  # 320 // 8 = 40
self.feat_w = self.img_w // self.downsample_factor  # 800 // 8 = 100
```

### Polar Coordinate Computation

Polar coordinates are computed relative to the global pole (vanishing point):

```python
# Center coordinates scaled to feature map space
center_h_feat = self.center_h / self.downsample_factor
center_w_feat = self.center_w / self.downsample_factor

# Compute polar coordinates for all feature map pixels
dx = x_coords - center_w_feat
dy = y_coords - center_h_feat
theta = np.arctan2(dy, dx)  # [-π, π]
r = np.sqrt(dx ** 2 + dy ** 2) * self.downsample_factor  # In image space
```

### Lane Mask Generation

Lanes are drawn at feature map resolution with appropriate thickness:

```python
# Scale lane coordinates to feature map resolution
lane_feat = lane / self.downsample_factor

# Draw with thickness ~2 at feature map scale (equivalent to ~16 at image scale)
cv2.line(lane_mask, pt1, pt2, 1, thickness=2)
```

## Usage

### Training

```python
python train.py --cfg Config/afplnet_culane_r18.py --save_path work_dir/afplnet
```

The training script will automatically:
1. Load AFPL-Net config
2. Build AFPL-Net model
3. Use AFPL-specific dataset (AFPLCULaneTrSet)
4. Generate appropriate ground truth
5. Train with AFPL loss functions

### Testing

```python
python test.py --cfg Config/afplnet_culane_r18.py --weight_path work_dir/afplnet/best.pth
```

The testing script will automatically:
1. Load AFPL-Net config
2. Build AFPL-Net model
3. Use AFPL-specific test dataset
4. Run inference and evaluation

## Verification

Three test files verify the implementation:

1. **`test_afplnet.py`**: Unit tests for AFPL-Net components
2. **`test_afpl_dataset.py`**: Tests for AFPL dataset structure
3. **`test_afpl_integration.py`**: Comprehensive integration tests

Run all tests:
```bash
python test_afplnet.py
python test_afpl_dataset.py
python test_afpl_integration.py
```

All tests should pass, confirming:
- ✓ Model architecture works
- ✓ Dataset generates correct format
- ✓ Training pipeline works
- ✓ Inference pipeline works
- ✓ Loss computation works

## Compatibility

The implementation maintains full backward compatibility:
- Polar R-CNN (two-stage) continues to use `BaseTrSet` from `base_dataset.py`
- AFPL-Net (single-stage) uses `AFPLBaseTrSet` from `afpl_base_dataset.py`
- The dataset builder automatically selects the appropriate dataset based on config

No changes are needed to existing Polar R-CNN code or configs.

## Future Extensions

To add AFPL support for other datasets (LLAMAS, TuSimple, etc.):

1. Create AFPL-specific dataset file (e.g., `afpl_llamas_dataset.py`)
2. Inherit from `AFPLBaseTrSet` and `AFPLBaseTsSet`
3. Implement dataset-specific `get_sample()` method
4. Update `build_trainset()` and `build_testset()` in `build.py`

Example:
```python
# In Dataset/afpl_llamas_dataset.py
from .afpl_base_dataset import AFPLBaseTrSet

class AFPLLLAMASTrSet(AFPLBaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms)
        # LLAMAS-specific initialization
        
    def get_sample(self, index):
        # LLAMAS-specific data loading
        pass
```

## Summary

The AFPL-Net data processing implementation provides:
- ✅ Single-stage specific ground truth generation
- ✅ Correct data format for AFPL-Net training
- ✅ Feature map resolution matching
- ✅ Backward compatibility with Polar R-CNN
- ✅ Extensible to other datasets
- ✅ Fully tested and verified
