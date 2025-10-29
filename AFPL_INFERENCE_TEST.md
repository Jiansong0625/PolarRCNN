# AFPL-Net Inference Test Documentation

## Overview

The `test_afplnet_inference.py` script is specifically designed for running inference with AFPL-Net (Anchor-Free Polar Lane Network). This script parallels the functionality of `test.py` but is adapted for AFPL-Net's single-stage, anchor-free architecture.

## Key Differences from test.py

### Architecture Support
- **test.py**: Designed for two-stage anchor-based Polar R-CNN
- **test_afplnet_inference.py**: Designed for single-stage anchor-free AFPL-Net

### Output Format
AFPL-Net produces lane detections differently from Polar R-CNN:
- **Polar R-CNN**: Uses RPN + ROI head pipeline with 20 predefined anchors
- **AFPL-Net**: Uses per-pixel predictions with angular clustering (NMS-free)

### Output Processing
The script includes a `format_afplnet_output()` function that converts AFPL-Net's output format to match the evaluator's expectations:
- Converts polar coordinates (θ, r) to Cartesian (x, y)
- Normalizes coordinates to [0, 1] range
- Sorts points by y-coordinate for proper interpolation
- Filters lanes with insufficient points

## Usage

### Basic Usage

```bash
python test_afplnet_inference.py \
    --cfg Config/afplnet_culane_r18.py \
    --weight_path path/to/weights.pth \
    --result_path ./results
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu_no` | 0 | GPU device ID |
| `--test_batch_size` | 32 | Batch size for testing |
| `--cfg` | ./Config/afplnet_culane_r18.py | Path to AFPL-Net config file |
| `--result_path` | ./result | Directory to save inference results |
| `--weight_path` | '' | Path to trained model weights |
| `--view_path` | ./view | Directory for visualizations |
| `--is_view` | 0 | Enable visualization mode (0 or 1) |
| `--is_val` | 0 | Use validation set instead of test set (0 or 1) |

### Testing on CULane Dataset

```bash
# Run inference and save results
python test_afplnet_inference.py \
    --cfg Config/afplnet_culane_r18.py \
    --weight_path work_dir/afplnet/best.pth \
    --result_path ./results/culane \
    --is_val 0

# The script will:
# 1. Load the AFPL-Net model
# 2. Load test images from the dataset
# 3. Run inference on all images
# 4. Save results in CULane format
# 5. Run evaluation metrics
```

### Visualization Mode

To visualize detected lanes instead of saving results:

```bash
python test_afplnet_inference.py \
    --cfg Config/afplnet_culane_r18.py \
    --weight_path work_dir/afplnet/best.pth \
    --view_path ./visualizations \
    --is_view 1
```

This will save images with detected lanes overlaid.

## Output Format

### Lane Detection Results

The script outputs lane detections in a format compatible with the CULane evaluator:

```python
{
    'lane_list': [  # List per batch
        [  # List of lanes for each image
            {
                'points': np.array([[x1, y1], [x2, y2], ...]),  # Normalized [0, 1]
                'conf': float  # Confidence score
            },
            ...
        ],
        ...
    ]
}
```

### File Structure

Results are saved in the same format as Polar R-CNN for compatibility:
```
result_path/
├── driver_23_30frame/
│   ├── 05140103_0449.MP4/
│   │   ├── 00000.lines.txt
│   │   ├── 00030.lines.txt
│   │   └── ...
│   └── ...
└── ...
```

Each `.lines.txt` file contains detected lanes in the format:
```
x1 y1 x2 y2 x3 y3 ...
x1 y1 x2 y2 x3 y3 ...
...
```

## Implementation Details

### AFPL-Net Output Processing

1. **Direct Lane Prediction**: Unlike Polar R-CNN, AFPL-Net directly predicts lanes during inference through its `post_process()` method:
   - Per-pixel classification and centerness scores are combined
   - Points above threshold are clustered by angle using DBSCAN
   - Each cluster represents one lane
   - Points are converted from polar to Cartesian coordinates

2. **Format Conversion**: The `format_afplnet_output()` function:
   ```python
   def format_afplnet_output(pred_dict, cfg):
       # Extract lanes from AFPL-Net output
       lanes_batch = pred_dict['lanes']
       
       # Convert to evaluator format
       for lanes in lanes_batch:
           for lane in lanes:
               # Normalize coordinates
               points[:, 0] = points[:, 0] / cfg.img_w
               points[:, 1] = points[:, 1] / cfg.img_h
               
               # Sort by y-coordinate for interpolation
               points = points[np.argsort(points[:, 1])]
   ```

3. **Evaluator Compatibility**: The formatted output is compatible with all existing evaluators (CULane, TuSimple, LLAMAS, etc.)

### Angular Clustering Post-Processing

AFPL-Net uses angular clustering instead of NMS:
- Points are clustered by their predicted angle θ
- No overlap-based NMS is needed
- Naturally separates lanes by their orientation
- More efficient than traditional NMS

## Comparison with test.py

| Feature | test.py | test_afplnet_inference.py |
|---------|---------|---------------------------|
| Model Type | Two-stage anchor-based | Single-stage anchor-free |
| Architecture | RPN + ROI Head | Direct prediction head |
| Post-processing | NMS-based | Angular clustering |
| Output Format | Via ROI head | Via post_process() |
| Anchors | 20 predefined | None (anchor-free) |

## Configuration Requirements

AFPL-Net configs must include:

```python
cfg_name = 'afplnet_culane_r18'  # Must contain 'afplnet'

# Global pole (vanishing point)
center_h = 25
center_w = 386

# Inference thresholds
conf_thres = 0.1
centerness_thres = 0.1

# Angular clustering
angle_cluster_eps = 0.035  # ~2 degrees
min_cluster_points = 10
```

## Troubleshooting

### "This script is designed for AFPL-Net configs"
- Make sure your config file name contains 'afplnet'
- If using Polar R-CNN, use `test.py` instead

### No lanes detected
- Check that model weights are loaded correctly
- Verify thresholds in config (conf_thres, centerness_thres)
- Try lowering thresholds if too strict

### Incorrect lane positions
- Verify that global pole (center_w, center_h) is set correctly
- Check if image preprocessing matches training

### Out of memory
- Reduce test_batch_size
- Use a smaller backbone (e.g., ResNet-18 instead of ResNet-50)

## Related Files

- `test.py` - Original inference script for Polar R-CNN
- `demo_afplnet.py` - Single image inference demo for AFPL-Net
- `test_afplnet.py` - Unit tests for AFPL-Net components
- `test_afpl_integration.py` - Integration tests for training/inference
- `Models/afpl_net.py` - AFPL-Net model definition
- `Models/Head/afpl_head.py` - AFPL detection head

## Example Workflow

```bash
# 1. Train AFPL-Net
python train.py --cfg Config/afplnet_culane_r18.py --save_path work_dir/afplnet

# 2. Run inference
python test_afplnet_inference.py \
    --cfg Config/afplnet_culane_r18.py \
    --weight_path work_dir/afplnet/best.pth

# 3. Results will be in ./result directory
# 4. Evaluation metrics will be printed to console
```

## Notes

- The script automatically detects if a config is for AFPL-Net
- All existing evaluators are supported (CULane, TuSimple, LLAMAS, CurveLanes, DLRail)
- Visualization mode creates images with lanes overlaid
- Results are saved in the same format as Polar R-CNN for consistency

## Future Improvements

Possible enhancements:
- Multi-GPU inference support
- TensorRT optimization
- ONNX export support
- Real-time video inference
- Custom visualization options
