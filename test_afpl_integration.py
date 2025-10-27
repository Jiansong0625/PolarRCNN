"""
Comprehensive integration test for AFPL-Net training and testing workflows

This test verifies the complete AFPL-Net pipeline without requiring actual dataset files.
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_training_workflow():
    """Test the complete training workflow"""
    print("\n" + "="*60)
    print("Testing AFPL-Net Training Workflow")
    print("="*60)
    
    try:
        import importlib.util
        
        # Load AFPL config
        spec = importlib.util.spec_from_file_location("config", "Config/afplnet_culane_r18.py")
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        print(f"✓ Config loaded: {cfg_module.cfg_name}")
        
        # Build model
        from Models.build import build_model
        model = build_model(cfg_module)
        print(f"✓ Model built: {type(model).__name__}")
        
        # Build loss function
        from Loss.overallloss import OverallLoss
        loss_fn = OverallLoss(cfg_module)
        print(f"✓ Loss function built")
        
        # Create mock training batch
        batch_size = 4
        feat_h = cfg_module.img_h // cfg_module.downsample_strides[0]
        feat_w = cfg_module.img_w // cfg_module.downsample_strides[0]
        
        sample_batch = {
            'img': torch.randn(batch_size, 3, cfg_module.img_h, cfg_module.img_w),
            'cls_gt': torch.randint(0, 2, (batch_size, feat_h, feat_w)).float(),
            'centerness_gt': torch.rand(batch_size, feat_h, feat_w),
            'theta_gt': (torch.rand(batch_size, feat_h, feat_w) - 0.5) * 2 * np.pi,
            'r_gt': torch.rand(batch_size, feat_h, feat_w) * 100,
        }
        print(f"✓ Mock batch created: batch_size={batch_size}")
        
        # Forward pass
        model.train()
        pred_dict = model(sample_batch)
        print(f"✓ Forward pass successful")
        
        # Check prediction shapes
        for key in ['cls_pred', 'centerness_pred', 'theta_pred', 'r_pred']:
            assert key in pred_dict, f"Missing prediction: {key}"
            expected_shape = (batch_size, 1, feat_h, feat_w)
            assert pred_dict[key].shape == expected_shape, f"Wrong shape for {key}: {pred_dict[key].shape}"
        print(f"✓ All predictions have correct shapes")
        
        # Compute loss
        loss, loss_msg = loss_fn(pred_dict, sample_batch)
        print(f"✓ Loss computed: {loss.item():.4f}")
        
        # Check loss components
        for key in ['loss_cls', 'loss_centerness', 'loss_regression', 'loss']:
            assert key in loss_msg, f"Missing loss component: {key}"
            assert isinstance(loss_msg[key], (int, float)), f"Loss component {key} is not a number"
        print(f"✓ All loss components present")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful")
        
        # Check that gradients exist
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                break
        assert has_grads, "No gradients computed"
        print(f"✓ Gradients computed")
        
        print("\n" + "="*60)
        print("✓ Training Workflow Test PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_workflow():
    """Test the complete inference workflow"""
    print("\n" + "="*60)
    print("Testing AFPL-Net Inference Workflow")
    print("="*60)
    
    try:
        import importlib.util
        
        # Load AFPL config
        spec = importlib.util.spec_from_file_location("config", "Config/afplnet_culane_r18.py")
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        print(f"✓ Config loaded: {cfg_module.cfg_name}")
        
        # Build model
        from Models.build import build_model
        model = build_model(cfg_module)
        print(f"✓ Model built: {type(model).__name__}")
        
        # Set to eval mode
        model.eval()
        print(f"✓ Model set to eval mode")
        
        # Create mock test batch
        batch_size = 2
        img_tensor = torch.randn(batch_size, 3, cfg_module.img_h, cfg_module.img_w)
        print(f"✓ Mock test batch created: batch_size={batch_size}")
        
        # Test get_lanes method
        with torch.no_grad():
            lanes = model.get_lanes(img_tensor)
        print(f"✓ get_lanes method successful")
        
        # Check output structure
        assert isinstance(lanes, list), "Output should be a list"
        assert len(lanes) == batch_size, f"Should have {batch_size} batch items, got {len(lanes)}"
        print(f"✓ Output structure correct: {len(lanes)} batches")
        
        # Check each batch
        for i, batch_lanes in enumerate(lanes):
            assert isinstance(batch_lanes, list), f"Batch {i} lanes should be a list"
            print(f"  Batch {i}: {len(batch_lanes)} lanes detected")
            
            # If lanes detected, check structure
            for j, lane in enumerate(batch_lanes[:2]):  # Check first 2 lanes
                assert isinstance(lane, dict), "Lane should be a dictionary"
                assert 'points' in lane, "Lane should have 'points' key"
                assert 'mean_score' in lane, "Lane should have 'mean_score' key"
                assert isinstance(lane['points'], list), "Lane points should be a list"
                assert isinstance(lane['mean_score'], (int, float, np.number)), "Lane score should be a number"
                print(f"    Lane {j}: {len(lane['points'])} points, score={lane['mean_score']:.3f}")
        
        print(f"✓ All detected lanes have correct structure")
        
        print("\n" + "="*60)
        print("✓ Inference Workflow Test PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_integration():
    """Test dataset integration with model and loss"""
    print("\n" + "="*60)
    print("Testing Dataset-Model-Loss Integration")
    print("="*60)
    
    try:
        import importlib.util
        from torchvision import transforms
        
        # Load AFPL config
        spec = importlib.util.spec_from_file_location("config", "Config/afplnet_culane_r18.py")
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        print(f"✓ Config loaded: {cfg_module.cfg_name}")
        
        # Create a mock dataset
        from Dataset.afpl_base_dataset import AFPLBaseTrSet
        
        class MockAFPLDataset(AFPLBaseTrSet):
            def __init__(self, cfg, transforms):
                super().__init__(cfg, transforms)
                self.img_path_list = ['dummy'] * 10
            
            def get_sample(self, index):
                # Generate dummy image and lanes
                img = np.random.randint(0, 255, (self.cfg.img_h, self.cfg.img_w, 3), dtype=np.uint8)
                
                # Create dummy lanes
                lanes = []
                for i in range(2):
                    x_start = 100 + i * 200
                    lane = np.array([
                        [x_start, 100], 
                        [x_start + 20, 150], 
                        [x_start + 40, 200], 
                        [x_start + 60, 250]
                    ], dtype=np.float32)
                    lanes.append(lane)
                
                return img, lanes
        
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = MockAFPLDataset(cfg_module, transforms=transform)
        print(f"✓ Mock dataset created: {len(dataset)} samples")
        
        # Test dataset output
        sample = dataset[0]
        print(f"✓ Dataset sample retrieved")
        
        # Check sample structure
        assert 'img' in sample, "Sample should have 'img'"
        assert 'cls_gt' in sample, "Sample should have 'cls_gt'"
        assert 'centerness_gt' in sample, "Sample should have 'centerness_gt'"
        assert 'theta_gt' in sample, "Sample should have 'theta_gt'"
        assert 'r_gt' in sample, "Sample should have 'r_gt'"
        print(f"✓ Sample has all required keys")
        
        # Create a batch
        batch = dataset.collate_fn([dataset[i] for i in range(4)])
        print(f"✓ Batch created: batch_size=4")
        
        # Test with model
        from Models.build import build_model
        model = build_model(cfg_module)
        model.train()
        
        pred_dict = model(batch)
        print(f"✓ Model forward pass with dataset batch successful")
        
        # Test with loss
        from Loss.overallloss import OverallLoss
        loss_fn = OverallLoss(cfg_module)
        
        loss, loss_msg = loss_fn(pred_dict, batch)
        print(f"✓ Loss computation with dataset batch successful: {loss.item():.4f}")
        
        print("\n" + "="*60)
        print("✓ Dataset-Model-Loss Integration Test PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    results = []
    
    results.append(("Training Workflow", test_training_workflow()))
    results.append(("Inference Workflow", test_inference_workflow()))
    results.append(("Dataset-Model-Loss Integration", test_dataset_integration()))
    
    # Print summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All integration tests passed! AFPL-Net is ready for training and testing.")
    
    sys.exit(0 if passed == total else 1)
