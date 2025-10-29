"""
Inference test script for AFPL-Net

This script is specifically designed for AFPL-Net (Anchor-Free Polar Lane Network)
and performs inference on test datasets, outputting results for evaluation.

Usage:
    python test_afplnet_inference.py --cfg Config/afplnet_culane_r18.py --weight_path path/to/weights.pth
"""

import torch
from tqdm import tqdm
from tools.get_config import get_cfg
from Dataset.build import build_testset
from Models.build import build_model
from Eval.build import build_evaluator
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='AFPL-Net Inference Test')
    parser.add_argument('--gpu_no', default=0, type=int, help='GPU device ID')
    parser.add_argument('--test_batch_size', default=32, type=int, help='Batch size for testing')
    parser.add_argument('--cfg', default='./Config/afplnet_culane_r18.py', type=str, help='Config file path')
    parser.add_argument('--result_path', default='./result', type=str, help='Path to save results')
    parser.add_argument('--weight_path', default='', type=str, help='Path to model weights')
    parser.add_argument('--view_path', default='./view', type=str, help='Path to save visualizations')
    parser.add_argument('--is_view', default=0, type=int, help='Whether to save visualizations (0 or 1)')
    parser.add_argument('--is_val', default=0, type=int, help='Whether to use validation set (0 or 1)')
    args = parser.parse_args()
    return args


def main():
    # Parse arguments and load config
    cfg = get_cfg(parse_args())
    torch.cuda.set_device(cfg.gpu_no)
    
    # Verify this is an AFPL-Net config
    if not hasattr(cfg, 'cfg_name') or 'afplnet' not in cfg.cfg_name.lower():
        print("Warning: This script is designed for AFPL-Net configs.")
        print(f"Current config: {getattr(cfg, 'cfg_name', 'Unknown')}")
        print("Consider using test.py for Polar R-CNN models.")
    
    # Build AFPL-Net model
    print("Building AFPL-Net model...")
    net = build_model(cfg)
    
    # Load weights
    if cfg.weight_path:
        print(f"Loading weights from: {cfg.weight_path}")
        net.load_state_dict(torch.load(cfg.weight_path, map_location='cpu'), strict=True)
    else:
        print("Warning: No weights specified, using random initialization")
    
    net.cuda().eval()
    print("Model loaded successfully!")
    
    # Build test dataset
    print("Building test dataset...")
    tsset = build_testset(cfg)
    print(f'Test set length: {len(tsset)}')
    
    # Build evaluator
    print("Building evaluator...")
    evaluator = build_evaluator(cfg)
    evaluator.pre_process()
    
    # Create data loader
    tsloader = torch.utils.data.DataLoader(
        tsset, 
        batch_size=cfg.test_batch_size, 
        shuffle=False, 
        num_workers=16,
        drop_last=False, 
        collate_fn=tsset.collate_fn
    )
    
    # Run inference
    print(f"Starting inference on {len(tsset)} images...")
    for i, (img, file_names, ori_imgs) in enumerate(tqdm(tsloader, desc='AFPL-Net inference')):
        with torch.no_grad():
            img = img.cuda()
            # Forward pass through AFPL-Net
            pred_dict = net(img)
            
            # AFPL-Net returns lanes directly in pred_dict
            # We need to format the output to match evaluator expectations
            outputs = format_afplnet_output(pred_dict, cfg)
        
        # Write or visualize output
        if cfg.is_view:
            evaluator.view_output(outputs, file_names, ori_imgs)
        else:
            evaluator.write_output(outputs, file_names)
    
    # Final evaluation or visualization
    if cfg.is_view:
        evaluator.view_gt()
    else:
        print("\nRunning evaluation...")
        evaluator.evaluate()
    
    print("\nInference completed!")


def format_afplnet_output(pred_dict, cfg):
    """
    Format AFPL-Net output to match evaluator expectations
    
    AFPL-Net outputs lanes directly during inference, so we need to
    convert them to the format expected by the evaluator.
    
    Args:
        pred_dict: Dictionary containing 'lanes' key from AFPL-Net
        cfg: Configuration object
        
    Returns:
        Dictionary with 'lane_list' key containing formatted lanes
    """
    import numpy as np
    
    lanes_batch = pred_dict['lanes']
    
    # Convert AFPL-Net lane format to evaluator format
    formatted_lanes = []
    for lanes in lanes_batch:
        batch_lanes = []
        for lane in lanes:
            # AFPL-Net lane: {'points': [(x, y), ...], 'scores': [...], 'mean_score': float}
            # Evaluator expects: {'points': np.array([[x, y], ...]), 'conf': float}
            
            points = np.array(lane['points'], dtype=np.float64)
            
            # Skip lanes with too few points
            if len(points) <= 1:
                continue
            
            # Normalize points to [0, 1] range
            # AFPL-Net returns points in pixel coordinates (img_w × img_h)
            points[:, 0] = points[:, 0] / cfg.img_w  # Normalize x
            points[:, 1] = points[:, 1] / cfg.img_h  # Normalize y
            
            # Clip to valid range [0, 1]
            points = np.clip(points, 0, 1)
            
            # Sort points by y-coordinate (top to bottom)
            # This is important for the interpolation in write_output_culane_format
            sort_idx = np.argsort(points[:, 1])
            points = points[sort_idx]
            
            formatted_lane = {
                'points': points,
                'conf': lane['mean_score']
            }
            batch_lanes.append(formatted_lane)
        formatted_lanes.append(batch_lanes)
    
    # Create output dictionary matching anchor-based model format
    outputs = {
        'lane_list': formatted_lanes
    }
    
    # For visualization, we need anchor_embeddings (not used in AFPL-Net)
    # Create dummy anchor embeddings for compatibility
    if cfg.is_view:
        batch_size = len(lanes_batch)
        outputs['anchor_embeddings'] = np.zeros((batch_size, 20, 2))  # Dummy anchors
    
    return outputs


if __name__ == '__main__':
    main()
