"""
Test AFPL-Net dataset implementation
Verifies that AFPL datasets generate correct ground truth format
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_afpl_dataset_structure():
    """Test that AFPL dataset generates correct data structure"""
    print("\n" + "="*60)
    print("Testing AFPL-Net Dataset Structure")
    print("="*60)
    
    try:
        from Dataset.afpl_base_dataset import AFPLBaseTrSet
        
        # Create a mock config
        class MockCfg:
            cfg_name = 'afplnet_test'
            img_h = 320
            img_w = 800
            center_h = 25
            center_w = 386
            max_lanes = 4
            random_seed = 3404
            train_augments = []
            downsample_strides = [8, 16, 32]
        
        cfg = MockCfg()
        feat_h = cfg.img_h // cfg.downsample_strides[0]
        feat_w = cfg.img_w // cfg.downsample_strides[0]
        
        # Create mock dataset
        class MockAFPLDataset(AFPLBaseTrSet):
            def __init__(self, cfg, transforms):
                super().__init__(cfg, transforms)
                self.img_path_list = ['dummy']
            
            def get_sample(self, index):
                # Generate dummy image and lanes
                img = np.random.randint(0, 255, (cfg.img_h, cfg.img_w, 3), dtype=np.uint8)
                
                # Create two dummy lanes
                lane1 = np.array([
                    [100, 100], [120, 150], [140, 200], [160, 250]
                ], dtype=np.float32)
                
                lane2 = np.array([
                    [300, 100], [320, 150], [340, 200], [360, 250]
                ], dtype=np.float32)
                
                lanes = [lane1, lane2]
                return img, lanes
        
        # Test dataset creation
        dataset = MockAFPLDataset(cfg, transforms=lambda x: torch.from_numpy(x).permute(2, 0, 1).float())
        print(f"✓ Dataset created successfully, length: {len(dataset)}")
        
        # Test data generation
        sample = dataset[0]
        
        # Check keys
        expected_keys = ['img', 'cls_gt', 'centerness_gt', 'theta_gt', 'r_gt']
        for key in expected_keys:
            assert key in sample, f"Missing key: {key}"
        print(f"✓ All expected keys present: {expected_keys}")
        
        # Check shapes
        assert sample['img'].shape == (3, cfg.img_h, cfg.img_w), f"Wrong img shape: {sample['img'].shape}"
        assert sample['cls_gt'].shape == (feat_h, feat_w), f"Wrong cls_gt shape: {sample['cls_gt'].shape}"
        assert sample['centerness_gt'].shape == (feat_h, feat_w), f"Wrong centerness_gt shape: {sample['centerness_gt'].shape}"
        assert sample['theta_gt'].shape == (feat_h, feat_w), f"Wrong theta_gt shape: {sample['theta_gt'].shape}"
        assert sample['r_gt'].shape == (feat_h, feat_w), f"Wrong r_gt shape: {sample['r_gt'].shape}"
        print("✓ All shapes correct")
        
        # Check value ranges
        assert sample['cls_gt'].min() >= 0 and sample['cls_gt'].max() <= 1, "cls_gt should be in [0, 1]"
        assert sample['centerness_gt'].min() >= 0 and sample['centerness_gt'].max() <= 1, "centerness_gt should be in [0, 1]"
        assert sample['theta_gt'].min() >= -np.pi and sample['theta_gt'].max() <= np.pi, "theta_gt should be in [-π, π]"
        assert sample['r_gt'].min() >= 0, "r_gt should be non-negative"
        print("✓ All value ranges correct")
        
        # Check that some lane pixels are marked
        num_lane_pixels = (sample['cls_gt'] > 0).sum()
        assert num_lane_pixels > 0, "No lane pixels marked in cls_gt"
        print(f"✓ Lane pixels marked: {num_lane_pixels}")
        
        # Test collate function
        batch = dataset.collate_fn([sample, sample])
        assert batch['img'].shape == (2, 3, cfg.img_h, cfg.img_w), f"Wrong batch img shape: {batch['img'].shape}"
        assert batch['cls_gt'].shape == (2, feat_h, feat_w), f"Wrong batch cls_gt shape: {batch['cls_gt'].shape}"
        print("✓ Batch collation works correctly")
        
        print("\n" + "="*60)
        print("✓ AFPL-Net Dataset Structure Test PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_builder():
    """Test that build_trainset works with AFPL config"""
    print("\n" + "="*60)
    print("Testing Dataset Builder with AFPL-Net")
    print("="*60)
    
    try:
        import importlib.util
        
        # Load AFPL config
        spec = importlib.util.spec_from_file_location("config", "Config/afplnet_culane_r18.py")
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        
        print(f"✓ Config loaded: {cfg_module.cfg_name}")
        print(f"  Dataset: {cfg_module.dataset}")
        
        # Check that it's recognized as AFPL config
        is_afpl = hasattr(cfg_module, 'cfg_name') and 'afplnet' in cfg_module.cfg_name.lower()
        assert is_afpl, "Config should be recognized as AFPL-Net config"
        print("✓ Config recognized as AFPL-Net config")
        
        # Note: We can't actually build the dataset without the CULane data files
        # But we can verify the import works
        from Dataset.build import build_trainset
        print("✓ build_trainset imported successfully")
        
        # Verify AFPL dataset classes can be imported
        from Dataset.afpl_culane_dataset import AFPLCULaneTrSet, AFPLCULaneTsSet
        print("✓ AFPL CULane dataset classes imported successfully")
        
        print("\n" + "="*60)
        print("✓ Dataset Builder Test PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    results = []
    
    results.append(("AFPL Dataset Structure", test_afpl_dataset_structure()))
    results.append(("Dataset Builder", test_dataset_builder()))
    
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
    
    sys.exit(0 if passed == total else 1)
