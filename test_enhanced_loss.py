"""
Test script for enhanced AFPL loss components

Tests the new loss functions added for optimization:
- Quality Focal Loss
- Polar IoU Loss  
- Uncertainty weighting
- Gradient normalization
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Loss.afpl_loss import (
    FocalLoss, QualityFocalLoss, CenternessLoss, 
    PolarRegressionLoss, PolarIoULoss, AFPLLoss
)


def test_quality_focal_loss():
    """Test Quality Focal Loss"""
    print("\n" + "="*60)
    print("Testing Quality Focal Loss")
    print("="*60)
    
    try:
        qfl = QualityFocalLoss(beta=2.0)
        
        # Create dummy data
        batch_size = 2
        h, w = 40, 100
        
        # Predictions (logits)
        pred = torch.randn(batch_size, 1, h, w)
        
        # Binary target
        target = torch.randint(0, 2, (batch_size, h, w)).float()
        
        # Quality target (centerness)
        quality_target = torch.rand(batch_size, h, w)
        
        # Test 1: Without quality target (fallback to binary)
        loss1 = qfl(pred, target)
        print(f"✓ Quality Focal Loss (binary mode): {loss1.item():.4f}")
        assert loss1.item() >= 0, "Loss should be non-negative"
        
        # Test 2: With quality target
        loss2 = qfl(pred, target, quality_target)
        print(f"✓ Quality Focal Loss (quality mode): {loss2.item():.4f}")
        assert loss2.item() >= 0, "Loss should be non-negative"
        
        # Test 3: Gradient flow
        pred.requires_grad = True
        loss3 = qfl(pred, target, quality_target)
        loss3.backward()
        assert pred.grad is not None, "Gradient should flow"
        print(f"✓ Gradient flow verified, grad norm: {pred.grad.norm().item():.4f}")
        
        print("✓ Quality Focal Loss: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Quality Focal Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_polar_iou_loss():
    """Test Polar IoU Loss"""
    print("\n" + "="*60)
    print("Testing Polar IoU Loss")
    print("="*60)
    
    try:
        # Test both IoU and GIoU
        for loss_type in ['iou', 'giou']:
            print(f"\nTesting {loss_type.upper()}...")
            
            polar_iou = PolarIoULoss(loss_type=loss_type)
            
            batch_size = 2
            h, w = 40, 100
            
            # Predictions
            theta_pred = torch.randn(batch_size, 1, h, w)
            r_pred = torch.abs(torch.randn(batch_size, 1, h, w)) + 1.0  # Positive radii
            
            # Targets
            theta_target = torch.randn(batch_size, h, w)
            r_target = torch.abs(torch.randn(batch_size, h, w)) + 1.0
            
            # Mask (some positive samples)
            mask = torch.randint(0, 2, (batch_size, h, w)).float()
            
            # Compute loss
            loss = polar_iou(theta_pred, r_pred, theta_target, r_target, mask)
            print(f"  ✓ {loss_type.upper()} Loss: {loss.item():.4f}")
            assert loss.item() >= 0, "Loss should be non-negative"
            assert loss.item() <= 2.0, "Loss should be bounded"
            
            # Test gradient flow
            theta_pred.requires_grad = True
            r_pred.requires_grad = True
            loss2 = polar_iou(theta_pred, r_pred, theta_target, r_target, mask)
            loss2.backward()
            assert theta_pred.grad is not None, "Gradient should flow to theta"
            assert r_pred.grad is not None, "Gradient should flow to r"
            print(f"  ✓ Gradient flow verified")
        
        # Test edge case: no positive samples
        print("\nTesting edge case: no positive samples...")
        mask_empty = torch.zeros(batch_size, h, w)
        loss_empty = polar_iou(theta_pred, r_pred, theta_target, r_target, mask_empty)
        assert loss_empty.item() == 0.0, "Loss should be 0 with no positive samples"
        print(f"  ✓ Empty mask handled correctly: {loss_empty.item():.4f}")
        
        # Test perfect prediction
        print("\nTesting perfect prediction...")
        theta_pred_perfect = theta_target.unsqueeze(1)
        r_pred_perfect = r_target.unsqueeze(1)
        mask_perfect = torch.ones(batch_size, h, w)
        loss_perfect = polar_iou(theta_pred_perfect, r_pred_perfect, theta_target, r_target, mask_perfect)
        print(f"  ✓ Perfect prediction loss: {loss_perfect.item():.4f} (should be close to 0)")
        assert loss_perfect.item() < 0.5, "Loss should be low for perfect predictions"
        
        print("\n✓ Polar IoU Loss: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Polar IoU Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uncertainty_weighting():
    """Test uncertainty-based weighting in PolarRegressionLoss"""
    print("\n" + "="*60)
    print("Testing Uncertainty Weighting")
    print("="*60)
    
    try:
        # Test with uncertainty enabled
        polar_loss_unc = PolarRegressionLoss(beta=1.0, use_uncertainty=True)
        
        # Test without uncertainty
        polar_loss_std = PolarRegressionLoss(beta=1.0, use_uncertainty=False)
        
        batch_size = 2
        h, w = 40, 100
        
        # Create data
        theta_pred = torch.randn(batch_size, 1, h, w)
        r_pred = torch.randn(batch_size, 1, h, w)
        theta_target = torch.randn(batch_size, h, w)
        r_target = torch.randn(batch_size, h, w)
        mask = torch.randint(0, 2, (batch_size, h, w)).float()
        
        # Compute losses
        loss_std = polar_loss_std(theta_pred, r_pred, theta_target, r_target, mask)
        loss_unc = polar_loss_unc(theta_pred, r_pred, theta_target, r_target, mask)
        
        print(f"✓ Standard loss: {loss_std.item():.4f}")
        print(f"✓ Uncertainty-weighted loss: {loss_unc.item():.4f}")
        
        # Check that uncertainty parameters exist and are learnable
        assert hasattr(polar_loss_unc, 'log_var_theta'), "Should have log_var_theta"
        assert hasattr(polar_loss_unc, 'log_var_r'), "Should have log_var_r"
        assert polar_loss_unc.log_var_theta.requires_grad, "log_var_theta should be learnable"
        assert polar_loss_unc.log_var_r.requires_grad, "log_var_r should be learnable"
        
        print(f"✓ Initial log_var_theta: {polar_loss_unc.log_var_theta.item():.4f}")
        print(f"✓ Initial log_var_r: {polar_loss_unc.log_var_r.item():.4f}")
        
        # Test that uncertainty parameters can be optimized
        optimizer = torch.optim.Adam(polar_loss_unc.parameters(), lr=0.01)
        initial_log_var = polar_loss_unc.log_var_theta.item()
        
        for i in range(5):
            optimizer.zero_grad()
            loss = polar_loss_unc(theta_pred, r_pred, theta_target, r_target, mask)
            loss.backward()
            optimizer.step()
        
        final_log_var = polar_loss_unc.log_var_theta.item()
        print(f"✓ After optimization, log_var_theta: {final_log_var:.4f}")
        
        print("✓ Uncertainty Weighting: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Uncertainty weighting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_normalization():
    """Test gradient normalization in AFPLLoss"""
    print("\n" + "="*60)
    print("Testing Gradient Normalization")
    print("="*60)
    
    try:
        # Create config
        class TestCfg:
            cls_loss_weight = 1.0
            cls_loss_alpha = 0.25
            cls_loss_gamma = 2.0
            centerness_loss_weight = 1.0
            regression_loss_weight = 2.0
            regression_beta = 1.0
            use_quality_focal = True
            use_polar_iou = True
            polar_iou_weight = 0.5
            polar_iou_type = 'iou'
            use_uncertainty = False
            use_grad_norm = True
        
        cfg = TestCfg()
        
        # Create loss with gradient normalization
        afpl_loss = AFPLLoss(cfg)
        afpl_loss.train()  # Set to training mode
        
        batch_size = 2
        h, w = 40, 100
        
        # Create dummy predictions
        pred_dict = {
            'cls_pred': torch.randn(batch_size, 1, h, w, requires_grad=True),
            'centerness_pred': torch.randn(batch_size, 1, h, w, requires_grad=True),
            'theta_pred': torch.randn(batch_size, 1, h, w, requires_grad=True),
            'r_pred': torch.randn(batch_size, 1, h, w, requires_grad=True)
        }
        
        # Create dummy targets
        target_dict = {
            'cls_gt': torch.randint(0, 2, (batch_size, h, w)).float(),
            'centerness_gt': torch.rand(batch_size, h, w),
            'theta_gt': torch.randn(batch_size, h, w),
            'r_gt': torch.abs(torch.randn(batch_size, h, w)) + 1.0
        }
        
        # Forward pass
        total_loss, loss_dict = afpl_loss(pred_dict, target_dict)
        
        print(f"✓ Total loss: {total_loss.item():.4f}")
        print(f"  - Classification loss: {loss_dict['loss_cls']:.4f}")
        print(f"  - Centerness loss: {loss_dict['loss_centerness']:.4f}")
        print(f"  - Regression loss: {loss_dict['loss_regression']:.4f}")
        print(f"  - Polar IoU loss: {loss_dict['loss_polar_iou']:.4f}")
        
        # Check gradient flow
        total_loss.backward()
        for key, tensor in pred_dict.items():
            assert tensor.grad is not None, f"Gradient should flow to {key}"
            print(f"  ✓ Gradient for {key}: norm={tensor.grad.norm().item():.4f}")
        
        # Test multiple iterations to see running mean updates
        print("\nTesting running mean updates...")
        initial_mean = afpl_loss.loss_cls_mean.item()
        
        for i in range(3):
            # Create new predictions
            pred_dict_new = {k: v.detach().clone().requires_grad_(True) for k, v in pred_dict.items()}
            total_loss, _ = afpl_loss(pred_dict_new, target_dict)
            total_loss.backward()
        
        final_mean = afpl_loss.loss_cls_mean.item()
        print(f"  ✓ Initial cls loss mean: {initial_mean:.4f}")
        print(f"  ✓ Final cls loss mean: {final_mean:.4f}")
        
        print("✓ Gradient Normalization: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Gradient normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_afpl_loss():
    """Test complete enhanced AFPL loss"""
    print("\n" + "="*60)
    print("Testing Enhanced AFPL Loss (Full Integration)")
    print("="*60)
    
    try:
        # Test different configurations
        configs = [
            ("Baseline (no enhancements)", {
                'use_quality_focal': False,
                'use_polar_iou': False,
                'use_uncertainty': False,
                'use_grad_norm': False
            }),
            ("Quality Focal only", {
                'use_quality_focal': True,
                'use_polar_iou': False,
                'use_uncertainty': False,
                'use_grad_norm': False
            }),
            ("Full enhancements", {
                'use_quality_focal': True,
                'use_polar_iou': True,
                'use_uncertainty': False,
                'use_grad_norm': True
            })
        ]
        
        batch_size = 2
        h, w = 40, 100
        
        # Create dummy data once
        pred_dict = {
            'cls_pred': torch.randn(batch_size, 1, h, w),
            'centerness_pred': torch.randn(batch_size, 1, h, w),
            'theta_pred': torch.randn(batch_size, 1, h, w),
            'r_pred': torch.randn(batch_size, 1, h, w)
        }
        
        target_dict = {
            'cls_gt': torch.randint(0, 2, (batch_size, h, w)).float(),
            'centerness_gt': torch.rand(batch_size, h, w),
            'theta_gt': torch.randn(batch_size, h, w),
            'r_gt': torch.abs(torch.randn(batch_size, h, w)) + 1.0
        }
        
        for config_name, config_params in configs:
            print(f"\n--- {config_name} ---")
            
            # Create config
            class TestCfg:
                cls_loss_weight = 1.0
                cls_loss_alpha = 0.25
                cls_loss_gamma = 2.0
                centerness_loss_weight = 1.0
                regression_loss_weight = 2.0
                regression_beta = 1.0
                polar_iou_weight = 0.5
                polar_iou_type = 'iou'
            
            cfg = TestCfg()
            for key, val in config_params.items():
                setattr(cfg, key, val)
            
            # Create loss
            afpl_loss = AFPLLoss(cfg)
            afpl_loss.train()
            
            # Forward pass
            total_loss, loss_dict = afpl_loss(pred_dict, target_dict)
            
            print(f"  Total loss: {total_loss.item():.4f}")
            for key, val in loss_dict.items():
                if key != 'loss':
                    print(f"    - {key}: {val:.4f}")
        
        print("\n✓ Enhanced AFPL Loss: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced AFPL Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests for enhanced loss components"""
    print("="*60)
    print("Enhanced AFPL Loss Component Tests")
    print("="*60)
    
    results = []
    
    results.append(("Quality Focal Loss", test_quality_focal_loss()))
    results.append(("Polar IoU Loss", test_polar_iou_loss()))
    results.append(("Uncertainty Weighting", test_uncertainty_weighting()))
    results.append(("Gradient Normalization", test_gradient_normalization()))
    results.append(("Enhanced AFPL Loss", test_enhanced_afpl_loss()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
