"""
Unit tests for AFPL-Net components

Tests the core functionality without requiring datasets or trained weights.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all AFPL-Net modules can be imported"""
    print("Testing imports...")
    
    try:
        from Models.afpl_net import AFPLNet
        print("✓ AFPLNet imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AFPLNet: {e}")
        return False
    
    try:
        from Models.Head.afpl_head import AFPLHead
        print("✓ AFPLHead imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AFPLHead: {e}")
        return False
    
    try:
        from Loss.afpl_loss import AFPLLoss, FocalLoss, CenternessLoss, PolarRegressionLoss
        print("✓ Loss functions imported successfully")
    except Exception as e:
        print(f"✗ Failed to import loss functions: {e}")
        return False
    
    return True


def test_config():
    """Test that config file is valid"""
    print("\nTesting configuration...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "Config/afplnet_culane_r18.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        # Check essential parameters
        assert hasattr(config, 'cfg_name'), "Missing cfg_name"
        assert hasattr(config, 'backbone'), "Missing backbone"
        assert hasattr(config, 'neck'), "Missing neck"
        assert hasattr(config, 'center_h'), "Missing center_h"
        assert hasattr(config, 'center_w'), "Missing center_w"
        assert hasattr(config, 'conf_thres'), "Missing conf_thres"
        assert hasattr(config, 'cls_loss_weight'), "Missing cls_loss_weight"
        
        print(f"✓ Config loaded: {config.cfg_name}")
        print(f"  - Backbone: {config.backbone}")
        print(f"  - Neck: {config.neck}")
        print(f"  - Global pole: ({config.center_w}, {config.center_h})")
        print(f"  - Confidence threshold: {config.conf_thres}")
        
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    return True


def test_model_structure():
    """Test AFPL-Net model structure"""
    print("\nTesting model structure...")
    
    try:
        import torch
        from Models.Head.afpl_head import AFPLHead
        
        # Create a dummy config
        class DummyCfg:
            img_w = 800
            img_h = 320
            neck_dim = 64
            center_w = 400
            center_h = 25
            conf_thres = 0.1
            centerness_thres = 0.1
            angle_cluster_eps = 0.035
            min_cluster_points = 10
        
        cfg = DummyCfg()
        
        # Create head
        head = AFPLHead(cfg)
        print("✓ AFPLHead created successfully")
        
        # Test forward pass with dummy input
        batch_size = 2
        dummy_feat = torch.randn(batch_size, cfg.neck_dim, cfg.img_h // 8, cfg.img_w // 8)
        
        pred_dict = head([dummy_feat])
        
        # Check outputs
        assert 'cls_pred' in pred_dict, "Missing cls_pred"
        assert 'centerness_pred' in pred_dict, "Missing centerness_pred"
        assert 'theta_pred' in pred_dict, "Missing theta_pred"
        assert 'r_pred' in pred_dict, "Missing r_pred"
        
        print(f"✓ Forward pass successful")
        print(f"  - cls_pred shape: {pred_dict['cls_pred'].shape}")
        print(f"  - centerness_pred shape: {pred_dict['centerness_pred'].shape}")
        print(f"  - theta_pred shape: {pred_dict['theta_pred'].shape}")
        print(f"  - r_pred shape: {pred_dict['r_pred'].shape}")
        
    except Exception as e:
        print(f"✗ Model structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_loss_functions():
    """Test loss functions"""
    print("\nTesting loss functions...")
    
    try:
        import torch
        from Loss.afpl_loss import FocalLoss, CenternessLoss, PolarRegressionLoss
        
        # Test Focal Loss
        focal_loss = FocalLoss()
        pred = torch.randn(2, 1, 40, 100)
        target = torch.randint(0, 2, (2, 40, 100)).float()
        loss = focal_loss(pred, target)
        assert loss.item() >= 0, "Focal loss should be non-negative"
        print(f"✓ Focal Loss: {loss.item():.4f}")
        
        # Test Centerness Loss
        centerness_loss = CenternessLoss()
        pred = torch.randn(2, 1, 40, 100)
        target = torch.rand(2, 40, 100)
        mask = torch.randint(0, 2, (2, 40, 100)).float()
        loss = centerness_loss(pred, target, mask)
        assert loss.item() >= 0, "Centerness loss should be non-negative"
        print(f"✓ Centerness Loss: {loss.item():.4f}")
        
        # Test Polar Regression Loss
        polar_loss = PolarRegressionLoss()
        theta_pred = torch.randn(2, 1, 40, 100)
        r_pred = torch.randn(2, 1, 40, 100)
        theta_target = torch.randn(2, 40, 100)
        r_target = torch.randn(2, 40, 100)
        mask = torch.randint(0, 2, (2, 40, 100)).float()
        loss = polar_loss(theta_pred, r_pred, theta_target, r_target, mask)
        assert loss.item() >= 0, "Polar regression loss should be non-negative"
        print(f"✓ Polar Regression Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_ground_truth_computation():
    """Test ground truth computation"""
    print("\nTesting ground truth computation...")
    
    try:
        import torch
        from Models.Head.afpl_head import AFPLHead
        
        class DummyCfg:
            img_w = 800
            img_h = 320
            neck_dim = 64
            center_w = 400
            center_h = 25
            conf_thres = 0.1
            centerness_thres = 0.1
            angle_cluster_eps = 0.035
            min_cluster_points = 10
        
        cfg = DummyCfg()
        head = AFPLHead(cfg)
        
        # Create dummy lane masks
        batch_size = 2
        lane_masks = torch.zeros(batch_size, cfg.img_h, cfg.img_w)
        # Add some lane pixels
        lane_masks[:, 150:160, 300:400] = 1.0
        
        # Compute ground truth
        gt_dict = head.compute_ground_truth(lane_masks)
        
        assert 'cls_gt' in gt_dict, "Missing cls_gt"
        assert 'centerness_gt' in gt_dict, "Missing centerness_gt"
        assert 'theta_gt' in gt_dict, "Missing theta_gt"
        assert 'r_gt' in gt_dict, "Missing r_gt"
        
        print("✓ Ground truth computation successful")
        print(f"  - cls_gt shape: {gt_dict['cls_gt'].shape}")
        print(f"  - Positive samples: {gt_dict['cls_gt'].sum().item()}")
        
    except Exception as e:
        print(f"✗ Ground truth test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("AFPL-Net Unit Tests")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    
    # Only run PyTorch-dependent tests if torch is available
    try:
        import torch
        results.append(("Model Structure", test_model_structure()))
        results.append(("Loss Functions", test_loss_functions()))
        results.append(("Ground Truth", test_ground_truth_computation()))
    except ImportError:
        print("\n⚠ PyTorch not installed, skipping model tests")
    
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
