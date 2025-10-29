# Test Script Comparison

This document helps you choose the correct test script for your model.

## Quick Reference

| Model Type | Test Script | Description |
|------------|-------------|-------------|
| **Polar R-CNN** | `test.py` | Two-stage anchor-based lane detector |
| **AFPL-Net** | `test_afplnet_inference.py` | Single-stage anchor-free lane detector |

## test.py

### When to Use
- You are using Polar R-CNN (two-stage, anchor-based model)
- Your config file does NOT contain 'afplnet' in the name
- Example configs: `polarrcnn_culane_r18.py`, `polarrcnn_tusimple_r18.py`

### Model Architecture
```
Input → Backbone → Neck → RPN Head → ROI Head → Lanes
                           (Anchors)   (Refinement)
```

### Output Format
Polar R-CNN outputs lanes through the ROI head:
```python
outputs = {
    'lane_list': [...],      # Detected lanes
    'anchor_embeddings': ... # Anchor parameters
}
```

### Usage Example
```bash
python test.py \
    --cfg Config/polarrcnn_culane_r18.py \
    --weight_path work_dir/polarrcnn/best.pth \
    --result_path ./results
```

## test_afplnet_inference.py

### When to Use
- You are using AFPL-Net (single-stage, anchor-free model)
- Your config file contains 'afplnet' in the name
- Example configs: `afplnet_culane_r18.py`

### Model Architecture
```
Input → Backbone → Neck → AFPL Head → Post-process → Lanes
                           (Per-pixel predictions + Angular clustering)
```

### Output Format
AFPL-Net outputs lanes through post-processing:
```python
pred_dict = {
    'cls_pred': ...,        # Classification predictions
    'centerness_pred': ..., # Centerness predictions
    'theta_pred': ...,      # Angle predictions
    'r_pred': ...,          # Radius predictions
    'lanes': [...]          # Post-processed lanes (inference only)
}
```

The `format_afplnet_output()` function converts this to evaluator format.

### Usage Example
```bash
python test_afplnet_inference.py \
    --cfg Config/afplnet_culane_r18.py \
    --weight_path work_dir/afplnet/best.pth \
    --result_path ./results
```

## Key Differences

### 1. Post-Processing
- **test.py**: Uses NMS (Non-Maximum Suppression) for duplicate removal
- **test_afplnet_inference.py**: Uses angular clustering (NMS-free)

### 2. Output Conversion
- **test.py**: Direct output from ROI head
- **test_afplnet_inference.py**: Requires `format_afplnet_output()` to convert

### 3. Anchors
- **test.py**: Works with 20 predefined anchors
- **test_afplnet_inference.py**: Anchor-free (per-pixel predictions)

### 4. Speed
- **test.py**: Slower (two-stage pipeline)
- **test_afplnet_inference.py**: Faster (single-stage)

## Common Arguments

Both scripts share these arguments:

```bash
--gpu_no 0                      # GPU device ID
--test_batch_size 32            # Batch size
--cfg <path>                    # Config file
--weight_path <path>            # Model weights
--result_path ./result          # Output directory
--is_view 0                     # Visualization mode
--is_val 0                      # Use validation set
```

## Troubleshooting

### Wrong Script for Model Type

**Symptom**: Error or incorrect output format

**Solution**: 
- Check your config file name
- If it contains 'afplnet', use `test_afplnet_inference.py`
- Otherwise, use `test.py`

### Script Detection Warning

If you use `test_afplnet_inference.py` with a non-AFPL-Net config:
```
Warning: This script is designed for AFPL-Net configs.
Current config: polarrcnn_culane_r18
Consider using test.py for Polar R-CNN models.
```

This means you should use `test.py` instead.

## Implementation Details

### test.py Flow
```python
net = build_model(cfg)          # Build Polar R-CNN
net.load_state_dict(...)        # Load weights
outputs = net(img)              # Forward pass
# outputs already has 'lane_list' key
evaluator.write_output(outputs, file_names)
```

### test_afplnet_inference.py Flow
```python
net = build_model(cfg)          # Build AFPL-Net
net.load_state_dict(...)        # Load weights
pred_dict = net(img)            # Forward pass (includes post_process)
# pred_dict has 'lanes' key, need to convert
outputs = format_afplnet_output(pred_dict, cfg)
# Now outputs has 'lane_list' key
evaluator.write_output(outputs, file_names)
```

## Summary

- Use **test.py** for Polar R-CNN (anchor-based)
- Use **test_afplnet_inference.py** for AFPL-Net (anchor-free)
- Both scripts produce compatible output for evaluation
- Both scripts work with all datasets (CULane, TuSimple, LLAMAS, etc.)

For more details:
- Polar R-CNN: See main [README.md](README.md)
- AFPL-Net: See [AFPLNET.md](AFPLNET.md) and [AFPL_INFERENCE_TEST.md](AFPL_INFERENCE_TEST.md)
