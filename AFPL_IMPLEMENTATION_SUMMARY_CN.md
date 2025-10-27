# AFPL-Net 数据处理完善总结 / AFPL-Net Data Processing Implementation Summary

## 问题描述 / Problem Description

afpl_net网络的数据处理中依然读取的是两阶段检测的参数，需要完善数据处理，使其能用afpl_net网络进行训练和测试。

The AFPL-Net network's data processing was still reading parameters for two-stage detection. The data processing needed to be improved so that the AFPL-Net network could be used for training and testing.

## 解决方案 / Solution

### 1. 创建AFPL专用数据集基类 / Created AFPL-specific Base Dataset Classes

**文件 / File:** `Dataset/afpl_base_dataset.py`

- 实现 `AFPLBaseTrSet` 训练数据集基类
- 实现 `AFPLBaseTsSet` 测试数据集基类
- **不需要** 两阶段检测参数（`num_line_groups`, `polar_map_size`）
- 生成特征图分辨率的ground truth（H/8 × W/8）
- 生成正确格式：`cls_gt`, `centerness_gt`, `theta_gt`, `r_gt`

Implemented:
- `AFPLBaseTrSet` training dataset base class
- `AFPLBaseTsSet` testing dataset base class
- Does NOT require two-stage parameters (`num_line_groups`, `polar_map_size`)
- Generates ground truth at feature map resolution (H/8 × W/8)
- Produces correct format: `cls_gt`, `centerness_gt`, `theta_gt`, `r_gt`

### 2. 实现CULane数据集 / Implemented CULane Dataset

**文件 / File:** `Dataset/afpl_culane_dataset.py`

- `AFPLCULaneTrSet`: CULane训练数据集
- `AFPLCULaneTsSet`: CULane测试数据集

Implemented:
- `AFPLCULaneTrSet`: CULane training dataset
- `AFPLCULaneTsSet`: CULane testing dataset

### 3. 更新数据集构建器 / Updated Dataset Builder

**文件 / File:** `Dataset/build.py`

- 自动检测AFPL-Net配置（检查 `cfg_name` 包含 'afplnet'）
- 对AFPL-Net使用专用数据集
- 对Polar R-CNN使用原有数据集
- 完全向后兼容

Modified:
- Automatically detects AFPL-Net config (checks if `cfg_name` contains 'afplnet')
- Uses AFPL-specific datasets for AFPL-Net
- Uses original datasets for Polar R-CNN
- Fully backward compatible

## 关键实现细节 / Key Implementation Details

### 特征图分辨率 / Feature Map Resolution

```python
# AFPL-Net的预测在特征图分辨率（下采样8倍）
# AFPL-Net predictions are at feature map resolution (downsampled 8x)
downsample_factor = 8
feat_h = img_h // downsample_factor  # 320 // 8 = 40
feat_w = img_w // downsample_factor  # 800 // 8 = 100
```

### 极坐标计算 / Polar Coordinate Computation

```python
# 相对于全局极点（消失点）计算极坐标
# Compute polar coordinates relative to global pole (vanishing point)
dx = x_coords - center_w_feat
dy = y_coords - center_h_feat
theta = np.arctan2(dy, dx)  # [-π, π]
r = np.sqrt(dx ** 2 + dy ** 2) * downsample_factor
```

## 测试验证 / Testing & Verification

### 测试文件 / Test Files

1. **`test_afplnet.py`** - AFPL-Net组件单元测试 / Component unit tests
2. **`test_afpl_dataset.py`** - 数据集结构测试 / Dataset structure tests  
3. **`test_afpl_integration.py`** - 完整流程集成测试 / Full pipeline integration tests

### 运行测试 / Run Tests

```bash
python test_afplnet.py          # ✓ 所有单元测试通过 / All unit tests pass
python test_afpl_dataset.py     # ✓ 数据集测试通过 / Dataset tests pass
python test_afpl_integration.py # ✓ 集成测试通过 / Integration tests pass
```

### 验证结果 / Verification Results

✅ 模型正确构建 / Model builds correctly
✅ 数据集类正确选择 / Dataset classes correctly selected
✅ 训练数据格式正确 / Training data format correct
✅ 前向传播工作 / Forward pass works
✅ 损失计算工作 / Loss computation works
✅ 反向传播和优化工作 / Backward pass and optimization work
✅ 推理工作 / Inference works

## 使用方法 / Usage

### 训练 / Training

```bash
python train.py --cfg Config/afplnet_culane_r18.py --save_path work_dir/afplnet
```

训练脚本将自动：
1. 加载AFPL-Net配置
2. 构建AFPL-Net模型
3. 使用AFPL专用数据集（AFPLCULaneTrSet）
4. 生成适当的ground truth
5. 使用AFPL损失函数训练

The training script will automatically:
1. Load AFPL-Net config
2. Build AFPL-Net model
3. Use AFPL-specific dataset (AFPLCULaneTrSet)
4. Generate appropriate ground truth
5. Train with AFPL loss functions

### 测试 / Testing

```bash
python test.py --cfg Config/afplnet_culane_r18.py --weight_path work_dir/afplnet/best.pth
```

测试脚本将自动：
1. 加载AFPL-Net配置
2. 构建AFPL-Net模型
3. 使用AFPL专用测试数据集
4. 运行推理和评估

The testing script will automatically:
1. Load AFPL-Net config
2. Build AFPL-Net model
3. Use AFPL-specific test dataset
4. Run inference and evaluation

## 兼容性 / Compatibility

### 完全向后兼容 / Fully Backward Compatible

- ✅ Polar R-CNN（两阶段）继续使用 `BaseTrSet`
- ✅ AFPL-Net（单阶段）使用 `AFPLBaseTrSet`
- ✅ 数据集构建器自动选择
- ✅ 无需修改现有Polar R-CNN代码或配置

- ✅ Polar R-CNN (two-stage) continues to use `BaseTrSet`
- ✅ AFPL-Net (single-stage) uses `AFPLBaseTrSet`
- ✅ Dataset builder automatically selects
- ✅ No changes needed to existing Polar R-CNN code or configs

## 文件清单 / File List

### 新增文件 / New Files

```
Dataset/afpl_base_dataset.py         - AFPL数据集基类 / AFPL base dataset classes
Dataset/afpl_culane_dataset.py       - CULane AFPL实现 / CULane AFPL implementation
test_afpl_dataset.py                 - 数据集测试 / Dataset tests
test_afpl_integration.py             - 集成测试 / Integration tests
AFPL_DATA_PROCESSING.md              - 实现文档 / Implementation documentation
AFPL_IMPLEMENTATION_SUMMARY_CN.md    - 中英文总结 / CN/EN summary
```

### 修改文件 / Modified Files

```
Dataset/build.py                     - 更新数据集构建逻辑 / Updated dataset builder logic
```

## 扩展到其他数据集 / Extending to Other Datasets

要为其他数据集（如LLAMAS、TuSimple）添加AFPL支持：

To add AFPL support for other datasets (e.g., LLAMAS, TuSimple):

```python
# 1. 创建AFPL专用数据集文件
# 1. Create AFPL-specific dataset file
# Dataset/afpl_llamas_dataset.py

from .afpl_base_dataset import AFPLBaseTrSet

class AFPLLLAMASTrSet(AFPLBaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms)
        # 数据集特定初始化 / Dataset-specific initialization
        
    def get_sample(self, index):
        # 数据集特定加载 / Dataset-specific loading
        pass

# 2. 更新 Dataset/build.py
# 2. Update Dataset/build.py
elif cfg.dataset == 'llamas':
    if is_afpl:
        from .afpl_llamas_dataset import AFPLLLAMASTrSet
        trainset = AFPLLLAMASTrSet(cfg=cfg, transforms=transform)
```

## 总结 / Summary

**问题 / Problem:** AFPL-Net使用两阶段检测的数据处理代码

**解决 / Solution:** 实现AFPL专用数据处理

**结果 / Result:** ✅ AFPL-Net现在可以用于训练和测试！

**Problem:** AFPL-Net was using data processing code for two-stage detection

**Solution:** Implemented AFPL-specific data processing

**Result:** ✅ AFPL-Net can now be used for training and testing!

---

## 更多信息 / More Information

详细实现说明请参考：`AFPL_DATA_PROCESSING.md`

For detailed implementation details, see: `AFPL_DATA_PROCESSING.md`
