# AFPL-Net: Anchor-Free Polar Lane Network

## Overview

AFPL-Net is a novel single-stage, anchor-free lane detection network that combines the best ideas from PolarMask and Polar R-CNN:

- **Global Pole Concept** (from Polar R-CNN): Uses a fixed vanishing point as a global reference for all lanes
- **Per-Pixel Prediction** (from PolarMask): Every feature map location independently predicts lane properties
- **Centerness for Quality** (from PolarMask): Predicts point quality to filter low-confidence predictions
- **Angular Clustering** (novel): NMS-free post-processing by grouping points with similar angles

## Key Innovations

### 1. Single-Stage, Anchor-Free Architecture
Unlike Polar R-CNN which has two stages (RPN for anchor proposals + ROI head for refinement), AFPL-Net directly predicts lane properties in a single forward pass. No need for 20 predefined anchor points!

### 2. Three Parallel Prediction Heads

Each feature map pixel predicts:

1. **Classification Score** (1 channel): Is this pixel part of any lane?
   - Loss: Focal Loss
   
2. **Centerness Score** (1 channel): How close is this pixel to the lane centerline?
   - Loss: Binary Cross Entropy (BCE)
   
3. **Polar Coordinates** (2 channels): (θ, r) relative to the global pole
   - Loss: Smooth L1

### 3. Angular Clustering (NMS-Free!)

The key insight: **lanes are naturally separated by their angles relative to the vanishing point**.

During post-processing:
1. Filter pixels by `classification_score × centerness_score`
2. Cluster remaining points by their predicted angle θ using DBSCAN
3. Each cluster = one lane
4. Sort points in each cluster by radius r
5. Convert polar coordinates back to Cartesian

**No NMS required!** Points with similar angles automatically belong to the same lane.

## Architecture

```
Input Image (H×W×3)
    ↓
Backbone (ResNet18/34/50 or DLA34)
    ↓
FPN Neck (Multi-scale feature fusion)
    ↓
AFPL Head (3 parallel branches)
    ├→ Classification Head → [B, 1, H', W']
    ├→ Centerness Head → [B, 1, H', W']
    └→ Polar Regression Head → [B, 2, H', W']
    ↓
Post-processing (Angular Clustering)
    ↓
Lane Curves
```

## Usage

### Training

```bash
python train.py --cfg Config/afplnet_culane_r18.py --save_path work_dir/afplnet
```

### Testing

```bash
python test.py --cfg Config/afplnet_culane_r18.py --weight_path work_dir/afplnet/best.pth
```

### Inference

```python
import torch
from Models.build import build_model
from tools.get_config import get_cfg

# Load config
class Args:
    cfg = 'Config/afplnet_culane_r18.py'
cfg = get_cfg(Args())

# Build model
model = build_model(cfg)
model.load_state_dict(torch.load('path/to/weights.pth'))
model.cuda().eval()

# Inference
img_tensor = ...  # [B, 3, H, W]
lanes = model.get_lanes(img_tensor)

# lanes is a list of detected lane curves
for lane in lanes[0]:  # First image in batch
    points = lane['points']  # List of (x, y) tuples
    score = lane['mean_score']  # Average confidence
```

## Configuration Parameters

Key parameters in config file:

```python
# Global pole (vanishing point)
center_h = 25  # y-coordinate
center_w = 386  # x-coordinate

# Inference thresholds
conf_thres = 0.1  # Classification threshold
centerness_thres = 0.1  # Centerness threshold

# Angular clustering
angle_cluster_eps = 0.035  # DBSCAN epsilon (~2 degrees)
min_cluster_points = 10  # Min points per lane

# Loss weights
cls_loss_weight = 1.0
centerness_loss_weight = 1.0
regression_loss_weight = 2.0
```

## Advantages over Polar R-CNN

1. **No Anchors**: No need to manually define 20 anchor points
2. **Single-Stage**: Simpler architecture, faster inference
3. **NMS-Free**: Angular clustering naturally separates lanes
4. **Per-Pixel**: Denser predictions, better for thin/occluded lanes

## Comparison with PolarMask

While AFPL-Net borrows ideas from PolarMask, there are key differences:

| Feature | PolarMask | AFPL-Net |
|---------|-----------|----------|
| Task | Instance Segmentation | Lane Detection |
| Pole | Object centroid (varies) | Global vanishing point (fixed) |
| Prediction | n rays from centroid | (θ, r) from global pole |
| Post-processing | NMS | Angular clustering (NMS-free) |

## File Structure

```
Models/
├── afpl_net.py              # Main AFPL-Net model
└── Head/
    └── afpl_head.py         # AFPL head with 3 prediction branches

Loss/
└── afpl_loss.py             # Loss functions (Focal, BCE, Smooth L1)

Config/
└── afplnet_culane_r18.py    # Example configuration
```

## Training from Scratch

1. Prepare dataset (CULane, Tusimple, etc.)
2. Create/modify config file
3. Run training:
```bash
python train.py --cfg Config/afplnet_culane_r18.py
```

The model will automatically:
- Use Focal Loss for handling class imbalance
- Use Centerness to improve point quality
- Learn polar coordinates relative to the global pole
- No need for anchor matching or NMS!

## Citation

If you use AFPL-Net, please cite:

```bibtex
@article{afplnet2024,
  title={AFPL-Net: Anchor-Free Polar Lane Network for Lane Detection},
  author={...},
  journal={...},
  year={2024}
}
```

And the papers that inspired this work:

- Polar R-CNN (for global pole concept)
- PolarMask (for anchor-free per-pixel prediction and centerness)
