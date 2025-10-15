# PolarRCNN F1 Score Improvements - Quick Start Guide

## 概述 (Overview)

这个PR包含了针对PolarRCNN网络的F1分数优化。所有改进都是通过调整配置文件中的超参数实现的，不涉及模型架构的修改。

This PR contains F1 score optimizations for the PolarRCNN network. All improvements are achieved through hyperparameter tuning in configuration files, with no model architecture changes.

## 主要改进 (Key Improvements)

### 1. 增强的数据增强 (Enhanced Data Augmentation)
- 更强的亮度、对比度和颜色变化
- 更多的几何变换（旋转、平移）
- 提高增强的应用概率

### 2. 优化的置信度阈值 (Optimized Confidence Thresholds)
- 降低阈值以提高召回率
- 平衡精确率和召回率

### 3. 重新平衡的损失函数 (Rebalanced Loss Functions)
- 增加IoU损失权重，提高定位精度
- 优化分类损失和Focal Loss参数
- 增强辅助损失的监督

### 4. 延长训练周期 (Extended Training)
- 增加训练epoch数，使模型更好地收敛

## 预期提升 (Expected Improvements)

| 数据集 Dataset | 当前 Current | 目标 Target | 提升 Improvement |
|---------------|-------------|------------|-----------------|
| TuSimple R18 | 97.94% | 98.2-98.5% | +0.26-0.56% |
| CULane R18 | 80.81% | 81.5-82.0% | +0.69-1.19% |
| CULane R34 | 80.92% | 81.6-82.1% | +0.68-1.18% |
| CULane R50 | 81.34% | 82.0-82.5% | +0.66-1.16% |
| CurveLanes DLA34 | 87.29% | 87.8-88.3% | +0.51-1.01% |
| LLAMAS R18 | 96.06% | 96.4-96.7% | +0.34-0.64% |
| LLAMAS DLA34 | 96.14% | 96.5-96.8% | +0.36-0.66% |
| DL-Rail R18 | 97.00% | 97.3-97.6% | +0.30-0.60% |

## 如何使用 (How to Use)

### 1. 训练新模型 (Train New Models)

使用改进的配置文件重新训练模型：

```bash
# TuSimple数据集 (TuSimple Dataset)
python train.py --cfg ./Config/polarrcnn_tusimple_r18.py --save_path work_dir/tusimple_improved

# CULane数据集 (CULane Dataset)
python train.py --cfg ./Config/polarrcnn_culane_r18.py --save_path work_dir/culane_improved

# 其他数据集 (Other Datasets)
python train.py --cfg ./Config/polarrcnn_<dataset>_<backbone>.py --save_path work_dir/<name>_improved
```

### 2. 评估模型 (Evaluate Models)

```bash
# 评估测试集 (Evaluate on test set)
python test.py --cfg ./Config/polarrcnn_<dataset>_<backbone>.py \
               --weight_path work_dir/<name>_improved/para_<epoch>.pth

# 可视化结果 (Visualize results)
python test.py --cfg ./Config/polarrcnn_<dataset>_<backbone>.py \
               --weight_path work_dir/<name>_improved/para_<epoch>.pth \
               --is_view 1
```

### 3. 监控训练过程 (Monitor Training)

建议使用TensorBoard监控训练过程：

```bash
# 启用TensorBoard (Enable TensorBoard)
python train.py --cfg ./Config/polarrcnn_<dataset>_<backbone>.py --use_tensorboard True

# 查看训练日志 (View training logs)
tensorboard --logdir=runs
```

## 重要提示 (Important Notes)

### ⚠️ 必须重新训练 (Retraining Required)
- 这些改进**需要重新训练模型**才能生效
- 旧的预训练权重不会自动获得这些提升
- The improvements **require retraining** models to take effect
- Old pretrained weights will not automatically benefit from these improvements

### 📊 验证集调优 (Validation Set Tuning)
- 建议在验证集上监控F1分数
- 可以根据验证集表现微调置信度阈值
- Monitor F1 scores on validation set
- Fine-tune confidence thresholds based on validation performance

### 💾 保存检查点 (Save Checkpoints)
- 定期保存训练检查点（当前每2个epoch保存一次）
- 选择验证集上F1分数最高的模型
- Save training checkpoints regularly (currently every 2 epochs)
- Choose the model with the highest validation F1 score

### 🔧 进一步优化 (Further Optimization)

如果需要进一步提升，可以考虑：
- 调整`conf_thres`在验证集上找到最佳值
- 尝试不同的`iou_loss_weight`值
- 增加更多的训练epoch（如果没有过拟合）
- 使用多个检查点的集成

For further improvements, consider:
- Tune `conf_thres` on validation set to find optimal value
- Try different `iou_loss_weight` values
- Add more training epochs (if no overfitting)
- Use ensemble of multiple checkpoints

## 技术细节 (Technical Details)

详细的技术文档：
- **F1_IMPROVEMENTS.md**: 改进策略和原理说明
- **CONFIGURATION_CHANGES.md**: 完整的参数对比表

Detailed technical documentation:
- **F1_IMPROVEMENTS.md**: Improvement strategy and rationale
- **CONFIGURATION_CHANGES.md**: Complete parameter comparison tables

## 文件修改清单 (Modified Files)

### 配置文件 (Configuration Files)
- `Config/polarrcnn_tusimple_r18.py`
- `Config/polarrcnn_culane_r18.py`
- `Config/polarrcnn_culane_r34.py`
- `Config/polarrcnn_culane_r50.py`
- `Config/polarrcnn_curvelanes_dla34.py`
- `Config/polarrcnn_llamas_r18.py`
- `Config/polarrcnn_llamas_dla34.py`
- `Config/polarrcnn_dlrail_r18.py`

### 新增文档 (New Documentation)
- `F1_IMPROVEMENTS.md` - 详细改进说明
- `CONFIGURATION_CHANGES.md` - 参数对比表
- `QUICKSTART.md` - 本文件
- `.gitignore` - Git忽略文件

## 联系方式 (Contact)

如有问题或建议，请在GitHub Issues中提出。

For questions or suggestions, please create a GitHub Issue.

## 致谢 (Acknowledgments)

这些改进基于：
- PolarRCNN原始论文的设计思想
- 车道检测领域的最佳实践
- 深度学习训练技巧和经验

These improvements are based on:
- Design principles from the original PolarRCNN paper
- Best practices in lane detection
- Deep learning training techniques and experience
