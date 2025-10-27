"""
Example script demonstrating AFPL-Net usage

This script shows how to:
1. Load the AFPL-Net model
2. Perform inference on an image
3. Visualize the detected lanes
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Models.build import build_model
from tools.get_config import get_cfg
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='AFPL-Net Demo')
    parser.add_argument('--cfg', default='Config/afplnet_culane_r18.py', type=str,
                        help='Config file for AFPL-Net')
    parser.add_argument('--weight', default='', type=str,
                        help='Path to model weights')
    parser.add_argument('--img', default='', type=str,
                        help='Path to input image')
    parser.add_argument('--output', default='demo_output.jpg', type=str,
                        help='Path to save output image')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU device ID')
    return parser.parse_args()


def load_model(cfg, weight_path, device='cuda'):
    """Load AFPL-Net model"""
    print("Building AFPL-Net model...")
    model = build_model(cfg)
    
    if weight_path:
        print(f"Loading weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def preprocess_image(img_path, target_size=(320, 800)):
    """Preprocess image for AFPL-Net"""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    ori_img = img.copy()
    
    # Resize
    img = cv2.resize(img, (target_size[1], target_size[0]))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize (ImageNet statistics)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # Convert to tensor [1, 3, H, W]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    
    return img, ori_img


def visualize_lanes(ori_img, lanes, output_path, cfg):
    """Visualize detected lanes on original image"""
    # Resize factors
    ori_h, ori_w = ori_img.shape[:2]
    model_h, model_w = cfg.img_h, cfg.img_w
    
    scale_x = ori_w / model_w
    scale_y = ori_h / model_h
    
    # Draw lanes
    vis_img = ori_img.copy()
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    print(f"\nDetected {len(lanes)} lanes:")
    for i, lane in enumerate(lanes):
        points = lane['points']
        score = lane['mean_score']
        color = colors[i % len(colors)]
        
        print(f"Lane {i+1}: {len(points)} points, score={score:.3f}")
        
        # Scale points to original image size
        scaled_points = []
        for x, y in points:
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_points.append((scaled_x, scaled_y))
        
        # Draw lane as connected line segments
        for j in range(len(scaled_points) - 1):
            pt1 = scaled_points[j]
            pt2 = scaled_points[j + 1]
            cv2.line(vis_img, pt1, pt2, color, 3)
        
        # Draw points
        for pt in scaled_points:
            cv2.circle(vis_img, pt, 3, color, -1)
    
    # Save output
    cv2.imwrite(output_path, vis_img)
    print(f"\nOutput saved to: {output_path}")
    
    return vis_img


def main():
    args = parse_args()
    
    # Load config
    print("Loading configuration...")
    
    class CfgArgs:
        def __init__(self):
            self.cfg = args.cfg
    
    cfg = get_cfg(CfgArgs())
    cfg.gpu_no = args.gpu
    
    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(cfg, args.weight, device)
    
    # Load and preprocess image
    print(f"\nLoading image: {args.img}")
    img_tensor, ori_img = preprocess_image(args.img, (cfg.img_h, cfg.img_w))
    img_tensor = img_tensor.to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        lanes = model.get_lanes(img_tensor)
    
    # Get lanes for first image in batch
    lanes = lanes[0] if len(lanes) > 0 else []
    
    if len(lanes) == 0:
        print("No lanes detected!")
        cv2.imwrite(args.output, ori_img)
    else:
        # Visualize results
        print("Visualizing results...")
        visualize_lanes(ori_img, lanes, args.output, cfg)
    
    print("\nDemo completed!")


if __name__ == '__main__':
    main()
