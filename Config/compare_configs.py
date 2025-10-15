"""
Configuration Comparison Script
Compares baseline and optimized configurations for PolarRCNN on CULane
"""

import sys
sys.path.insert(0, '.')

def count_model_params(cfg_module):
    """Calculate approximate model parameters based on configuration"""
    # Backbone params (ResNet18): ~11.2M
    backbone_params = 11.2e6
    
    # Neck params (FPN)
    neck_dim = cfg_module.neck_dim
    fpn_channels = cfg_module.fpn_in_channel
    neck_params = 0
    for in_ch in fpn_channels:
        # lateral conv: in_ch * neck_dim * 1 * 1
        neck_params += in_ch * neck_dim
        # fpn conv: neck_dim * neck_dim * 3 * 3
        neck_params += neck_dim * neck_dim * 9
    
    # RPN Head params
    rpn_params = neck_dim * 64 + 64 * 1  # cls layers
    rpn_params += neck_dim * 2  # reg layers
    
    # RoI Head params
    fc_input = cfg_module.num_feat_samples * cfg_module.prior_feat_channels
    fc_hidden = cfg_module.fc_hidden_dim
    gnn_dim = cfg_module.gnn_inter_dim
    num_offsets = cfg_module.num_offsets
    num_line_groups = cfg_module.num_line_groups
    
    # FC layer
    roi_params = fc_input * fc_hidden
    # Cls block
    roi_params += fc_hidden * fc_hidden + fc_hidden * 1
    # Reg block  
    roi_params += fc_hidden * fc_hidden + fc_hidden * (num_offsets + 2)
    # Aux reg
    roi_params += fc_hidden * (2 * num_line_groups)
    # GNN
    roi_params += 2 * gnn_dim + fc_hidden * gnn_dim * 3
    roi_params += gnn_dim * gnn_dim + gnn_dim * cfg_module.iou_dim
    roi_params += cfg_module.iou_dim * fc_hidden + fc_hidden * fc_hidden
    
    total_params = backbone_params + neck_params + rpn_params + roi_params
    return total_params

def print_comparison():
    print("=" * 80)
    print("PolarRCNN Configuration Comparison: Baseline vs Optimized")
    print("=" * 80)
    print()
    
    # Import configurations
    from Config import polarrcnn_culane_r18 as baseline
    from Config import polarrcnn_culane_r18_optimized as optimized
    
    # Calculate parameters
    baseline_params = count_model_params(baseline)
    optimized_params = count_model_params(optimized)
    param_reduction = (baseline_params - optimized_params) / baseline_params * 100
    
    print("MODEL ARCHITECTURE")
    print("-" * 80)
    print(f"{'Parameter':<30} {'Baseline':<20} {'Optimized':<20} {'Change':<10}")
    print("-" * 80)
    print(f"{'Backbone':<30} {'ResNet18':<20} {'ResNet18':<20} {'-':<10}")
    print(f"{'Neck Dimension':<30} {baseline.neck_dim:<20} {optimized.neck_dim:<20} {f'{(optimized.neck_dim-baseline.neck_dim)/baseline.neck_dim*100:+.1f}%':<10}")
    print(f"{'FC Hidden Dimension':<30} {baseline.fc_hidden_dim:<20} {optimized.fc_hidden_dim:<20} {f'{(optimized.fc_hidden_dim-baseline.fc_hidden_dim)/baseline.fc_hidden_dim*100:+.1f}%':<10}")
    print(f"{'GNN Inter Dimension':<30} {baseline.gnn_inter_dim:<20} {optimized.gnn_inter_dim:<20} {f'{(optimized.gnn_inter_dim-baseline.gnn_inter_dim)/baseline.gnn_inter_dim*100:+.1f}%':<10}")
    print(f"{'Prior Feat Channels':<30} {baseline.prior_feat_channels:<20} {optimized.prior_feat_channels:<20} {f'{(optimized.prior_feat_channels-baseline.prior_feat_channels)/baseline.prior_feat_channels*100:+.1f}%':<10}")
    print(f"{'Total Parameters (M)':<30} {baseline_params/1e6:<20.2f} {optimized_params/1e6:<20.2f} {f'{-param_reduction:+.1f}%':<10}")
    print()
    
    print("TRAINING CONFIGURATION")
    print("-" * 80)
    print(f"{'Parameter':<30} {'Baseline':<20} {'Optimized':<20} {'Change':<10}")
    print("-" * 80)
    print(f"{'Batch Size':<30} {baseline.batch_size:<20} {optimized.batch_size:<20} {'-':<10}")
    print(f"{'Epochs':<30} {baseline.epoch_num:<20} {optimized.epoch_num:<20} {f'{(optimized.epoch_num-baseline.epoch_num)/baseline.epoch_num*100:+.1f}%':<10}")
    print(f"{'Learning Rate':<30} {baseline.lr:<20} {optimized.lr:<20} {'-':<10}")
    print(f"{'Warmup Iterations':<30} {baseline.warmup_iter:<20} {optimized.warmup_iter:<20} {'-':<10}")
    print()
    
    print("LOSS CONFIGURATION")
    print("-" * 80)
    print(f"{'Parameter':<30} {'Baseline':<20} {'Optimized':<20} {'Change':<10}")
    print("-" * 80)
    print(f"{'Cls Loss Alpha':<30} {baseline.cls_loss_alpha:<20.3f} {optimized.cls_loss_alpha:<20.3f} {f'{(optimized.cls_loss_alpha-baseline.cls_loss_alpha)/baseline.cls_loss_alpha*100:+.1f}%':<10}")
    print(f"{'Cls Loss Alpha O2O':<30} {baseline.cls_loss_alpha_o2o:<20.3f} {optimized.cls_loss_alpha_o2o:<20.3f} {f'{(optimized.cls_loss_alpha_o2o-baseline.cls_loss_alpha_o2o)/baseline.cls_loss_alpha_o2o*100:+.1f}%':<10}")
    print(f"{'Rank Loss Weight':<30} {baseline.rank_loss_weight:<20.2f} {optimized.rank_loss_weight:<20.2f} {f'{(optimized.rank_loss_weight-baseline.rank_loss_weight)/baseline.rank_loss_weight*100:+.1f}%':<10}")
    print()
    
    print("INFERENCE CONFIGURATION")
    print("-" * 80)
    print(f"{'Parameter':<30} {'Baseline':<20} {'Optimized':<20} {'Change':<10}")
    print("-" * 80)
    print(f"{'Confidence Threshold':<30} {baseline.conf_thres:<20.3f} {optimized.conf_thres:<20.3f} {f'{(optimized.conf_thres-baseline.conf_thres)/baseline.conf_thres*100:+.1f}%':<10}")
    print(f"{'Conf Threshold NMS-free':<30} {baseline.conf_thres_nmsfree:<20.3f} {optimized.conf_thres_nmsfree:<20.3f} {f'{(optimized.conf_thres_nmsfree-baseline.conf_thres_nmsfree)/baseline.conf_thres_nmsfree*100:+.1f}%':<10}")
    print()
    
    print("DATA AUGMENTATION")
    print("-" * 80)
    print(f"{'Baseline Augmentations:':<30} {len(baseline.train_augments)} transforms")
    print(f"{'Optimized Augmentations:':<30} {len(optimized.train_augments)} transforms")
    print()
    print("New augmentations in optimized config:")
    print("  - CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    print("  - Gaussian Noise")
    print("  - Enhanced brightness range: (-0.15, 0.15) -> (-0.25, 0.25)")
    print("  - Added contrast augmentation: (-0.15, 0.15)")
    print("  - Enhanced HSV saturation and value shifts")
    print()
    
    print("EXPECTED IMPROVEMENTS")
    print("-" * 80)
    print("✓ Model Size: ~{:.1f}% reduction in parameters".format(param_reduction))
    print("✓ Inference Speed: ~10-15% faster (estimated)")
    print("✓ Memory Usage: ~20% reduction (estimated)")
    print("✓ Night Scene F1: +1-2 points improvement (estimated)")
    print("✓ Overall F1@50: +0.3-0.7 points improvement (estimated)")
    print()
    print("=" * 80)

if __name__ == "__main__":
    print_comparison()
