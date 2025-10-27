# AFPL-Net Implementation Summary

## What Was Implemented

This implementation adds **AFPL-Net (Anchor-Free Polar Lane Network)** to the PolarRCNN repository. AFPL-Net is a novel single-stage, anchor-free lane detection model that combines ideas from PolarMask and Polar R-CNN.

## Key Components

### 1. Model Architecture (`Models/afpl_net.py`, `Models/Head/afpl_head.py`)

**AFPLNet**: Single-stage lane detection model
- Backbone: ResNet/DLA (reuses existing)
- Neck: FPN (reuses existing)
- Head: New AFPL head with 3 parallel branches

**AFPLHead**: Three parallel prediction branches
- **Classification Head** (H×W×1): Predicts if pixel is on a lane
- **Centerness Head** (H×W×1): Predicts point quality (distance to centerline)
- **Polar Regression Head** (H×W×2): Predicts (θ, r) relative to global pole

### 2. Loss Functions (`Loss/afpl_loss.py`)

- **Focal Loss**: For classification (handles class imbalance)
- **BCE Loss**: For centerness (point quality estimation)
- **Smooth L1 Loss**: For polar regression (θ, r)
- **AFPLLoss**: Combines all three with configurable weights

### 3. Post-Processing (in `Models/Head/afpl_head.py`)

**Angular Clustering** - The key innovation!
- Groups points by predicted angle θ using DBSCAN
- Points with similar angles → same lane
- **No NMS needed!**
- Sorts points by radius r to form lane curves

### 4. Configuration (`Config/afplnet_culane_r18.py`)

Example config for CULane dataset with:
- Global pole position (vanishing point)
- Loss weights and hyperparameters
- Inference thresholds
- Angular clustering parameters

### 5. Utilities

- **demo_afplnet.py**: Demo script for inference on images
- **test_afplnet.py**: Unit tests for validation
- **.gitignore**: Properly ignores build artifacts
- **AFPLNET.md**: Comprehensive documentation

## How It Works

### Training Flow
```
Input Image → Backbone → FPN → AFPL Head → 3 Predictions
                                            ├─ Classification
                                            ├─ Centerness  
                                            └─ Polar (θ,r)
                                            ↓
                                        Compute Losses
                                            ├─ Focal Loss
                                            ├─ BCE Loss
                                            └─ Smooth L1
                                            ↓
                                        Backpropagate
```

### Inference Flow
```
Input Image → Model Forward → Predictions
                              ↓
                    Filter by score > threshold
                              ↓
                    Angular Clustering (DBSCAN)
                              ↓
                    Group by similar θ
                              ↓
                    Sort each group by r
                              ↓
                    Convert polar → Cartesian
                              ↓
                    Lane Curves
```

## Key Innovations

### 1. Anchor-Free
- **Problem**: Polar R-CNN needs 20 predefined anchors
- **Solution**: AFPL-Net predicts from every pixel, no anchors needed

### 2. Single-Stage
- **Problem**: Polar R-CNN has 2 stages (RPN + ROI)
- **Solution**: AFPL-Net directly predicts in one stage

### 3. NMS-Free
- **Problem**: Traditional methods need NMS to remove duplicates
- **Solution**: Angular clustering naturally separates lanes

### 4. Global Pole
- **From Polar R-CNN**: Uses fixed vanishing point
- **Advantage**: Stable reference for all lanes

### 5. Centerness
- **From PolarMask**: Predicts point quality
- **Advantage**: Filters low-confidence predictions

## Comparison with Polar R-CNN

| Feature | Polar R-CNN | AFPL-Net |
|---------|-------------|----------|
| Stages | 2 (RPN + ROI) | 1 (Direct) |
| Anchors | 20 predefined | 0 (anchor-free) |
| Prediction | Sparse (20 locations) | Dense (every pixel) |
| Post-processing | NMS or GNN | Angular clustering |
| Architecture | More complex | Simpler |

## Usage Examples

### Training
```bash
python train.py --cfg Config/afplnet_culane_r18.py --save_path work_dir/afplnet
```

### Testing
```bash
python test.py --cfg Config/afplnet_culane_r18.py --weight_path work_dir/afplnet/best.pth
```

### Demo
```bash
python demo_afplnet.py --cfg Config/afplnet_culane_r18.py --weight weights.pth --img test.jpg
```

## Files Added/Modified

### New Files (8 files)
1. `Models/afpl_net.py` - Main model
2. `Models/Head/afpl_head.py` - Detection head
3. `Loss/afpl_loss.py` - Loss functions
4. `Config/afplnet_culane_r18.py` - Config
5. `AFPLNET.md` - Documentation
6. `demo_afplnet.py` - Demo script
7. `test_afplnet.py` - Unit tests
8. `.gitignore` - Git ignore file

### Modified Files (3 files)
1. `Models/build.py` - Register AFPL-Net
2. `Loss/overallloss.py` - Add AFPL loss
3. `README.md` - Add AFPL-Net section

### Removed
- All `__pycache__` files (now properly ignored)

## Integration Points

The implementation cleanly integrates with existing code:

1. **Backbone**: Reuses existing ResNet/DLA backbones
2. **Neck**: Reuses existing FPN
3. **Build System**: Automatic detection based on config name
4. **Training Loop**: Compatible with existing training script
5. **Evaluation**: Compatible with existing evaluation framework

## Next Steps

To use AFPL-Net:

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare dataset (CULane, etc.)
3. Train: `python train.py --cfg Config/afplnet_culane_r18.py`
4. Test: `python test.py --cfg Config/afplnet_culane_r18.py --weight path/to/weights.pth`

## References

This implementation was inspired by:
- **Polar R-CNN**: Global pole concept for lane detection
- **PolarMask**: Anchor-free single-stage instance segmentation
- **FCOS**: Per-pixel prediction and centerness

The key innovation is combining these ideas specifically for lane detection with angular clustering for NMS-free post-processing.
