#!/usr/bin/env python3
"""
演示极点参考系不一致问题和解决方案
Demonstration of the pole reference system inconsistency issue and solution
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Circle
import os

def draw_coordinate_system(ax, origin, img_w, img_h, title, lanes=None, pole=None, 
                          show_polar=False, lane_color='blue', pole_color='red'):
    """绘制坐标系统和车道"""
    # 绘制图像边界
    ax.set_xlim(-50, img_w + 50)
    ax.set_ylim(-50, img_h + 50)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 图像坐标系y轴向下
    
    # 绘制边框
    ax.plot([0, img_w, img_w, 0, 0], [0, 0, img_h, img_h, 0], 'k-', linewidth=2)
    
    # 绘制极点
    if pole is not None:
        ax.plot(pole[0], pole[1], 'o', color=pole_color, markersize=12, 
                label=f'极点 ({pole[0]:.0f}, {pole[1]:.0f})', zorder=5)
        
        # 绘制极点的十字标记
        ax.plot([pole[0]-10, pole[0]+10], [pole[1], pole[1]], 
                color=pole_color, linewidth=2, zorder=4)
        ax.plot([pole[0], pole[0]], [pole[1]-10, pole[1]+10], 
                color=pole_color, linewidth=2, zorder=4)
    
    # 绘制车道
    if lanes is not None:
        for i, lane in enumerate(lanes):
            ax.plot(lane[:, 0], lane[:, 1], 'o-', color=lane_color, 
                   linewidth=2, markersize=6, label=f'车道{i+1}' if i < 2 else '')
            
            # 如果显示极坐标，绘制从极点到车道中点的连线
            if show_polar and pole is not None:
                mid_idx = len(lane) // 2
                mid_point = lane[mid_idx]
                
                # 绘制从极点到车道点的箭头
                arrow = FancyArrowPatch(
                    (pole[0], pole[1]), (mid_point[0], mid_point[1]),
                    arrowstyle='->', mutation_scale=20, linewidth=1.5,
                    color='green', linestyle='--', alpha=0.7
                )
                ax.add_patch(arrow)
                
                # 计算并显示极坐标
                dx = mid_point[0] - pole[0]
                dy = pole[1] - mid_point[1]  # 注意y轴方向
                theta = np.arctan2(dy, dx)
                r = np.sqrt(dx**2 + dy**2)
                
                # 在车道点附近显示极坐标值
                ax.text(mid_point[0] + 20, mid_point[1], 
                       f'θ={theta:.2f}\nr={r:.0f}',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('x (像素)', fontsize=10)
    ax.set_ylabel('y (像素)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)


def demonstrate_horizontal_flip_issue():
    """演示水平翻转时的极点参考系问题"""
    
    # 图像尺寸和极点设置
    img_w, img_h = 800, 320
    original_pole = np.array([400, 25])  # 原始极点（接近消失点）
    
    # 创建一个简单的车道（左侧车道）
    lane1 = np.array([
        [150, 250],
        [160, 200],
        [170, 150],
        [180, 100],
        [190, 50]
    ], dtype=np.float32)
    
    lane2 = np.array([
        [250, 250],
        [260, 200],
        [270, 150],
        [280, 100],
        [290, 50]
    ], dtype=np.float32)
    
    lanes = [lane1, lane2]
    
    # 水平翻转变换
    flipped_lanes = []
    for lane in lanes:
        flipped_lane = lane.copy()
        flipped_lane[:, 0] = img_w - lane[:, 0]  # x' = img_w - x
        flipped_lanes.append(flipped_lane)
    
    # 正确的极点翻转
    correct_flipped_pole = np.array([img_w - original_pole[0], original_pole[1]])
    
    # 错误：极点不翻转
    wrong_pole = original_pole.copy()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('水平翻转时的极点参考系问题演示', fontsize=16, fontweight='bold')
    
    # 1. 原始图像
    draw_coordinate_system(
        axes[0, 0], (0, 0), img_w, img_h,
        '1. 原始图像',
        lanes=lanes, pole=original_pole, show_polar=True,
        lane_color='blue', pole_color='red'
    )
    
    # 2. 翻转后的图像（车道翻转，极点未翻转 - 错误）
    draw_coordinate_system(
        axes[0, 1], (0, 0), img_w, img_h,
        '2. 翻转后（错误：极点未翻转）',
        lanes=flipped_lanes, pole=wrong_pole, show_polar=True,
        lane_color='orange', pole_color='red'
    )
    
    # 添加错误标记
    axes[0, 1].text(img_w/2, img_h - 30, '❌ 极点位置错误！', 
                   fontsize=14, color='red', fontweight='bold',
                   ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 3. 翻转后的图像（车道和极点都翻转 - 正确）
    draw_coordinate_system(
        axes[1, 0], (0, 0), img_w, img_h,
        '3. 翻转后（正确：极点同步翻转）',
        lanes=flipped_lanes, pole=correct_flipped_pole, show_polar=True,
        lane_color='green', pole_color='darkgreen'
    )
    
    # 添加正确标记
    axes[1, 0].text(img_w/2, img_h - 30, '✓ 极点正确翻转！', 
                   fontsize=14, color='green', fontweight='bold',
                   ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 4. 对比说明
    axes[1, 1].axis('off')
    explanation_text = """
    问题说明：
    
    原始图像：
    • 极点坐标：(400, 25)
    • 车道在左侧
    • 极坐标(θ, r)基于极点(400, 25)计算
    
    水平翻转后：
    
    ❌ 错误方式（图2）：
    • 极点仍为：(400, 25) - 未翻转
    • 车道翻转到右侧
    • 极坐标仍基于旧极点(400, 25)计算
    • 结果：θ和r的值完全错误！
    
    ✓ 正确方式（图3）：
    • 极点变为：(400, 25) - 同步翻转
    • 车道翻转到右侧
    • 极坐标基于新极点(400, 25)计算
    • 结果：θ和r的值保持几何一致性！
    
    关键点：
    极点必须与图像和车道点一起变换，
    才能保证极坐标ground truth的正确性。
    
    解决方案：
    将极点作为关键点加入albumentations
    的变换流程，自动同步变换。
    """
    
    axes[1, 1].text(0.1, 0.9, explanation_text, 
                   transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top',
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = '/home/runner/work/PolarRCNN/PolarRCNN'
    output_path = os.path.join(output_dir, 'pole_reference_issue_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 演示图已保存到: {output_path}")
    plt.close()


def demonstrate_affine_transform_issue():
    """演示仿射变换时的极点参考系问题"""
    
    # 图像尺寸和极点设置
    img_w, img_h = 800, 320
    original_pole = np.array([400, 25])
    
    # 创建车道
    lane = np.array([
        [200, 250],
        [210, 200],
        [220, 150],
        [230, 100],
        [240, 50]
    ], dtype=np.float32)
    
    # 简单的平移变换（向右平移80像素，向下平移20像素）
    tx, ty = 80, 20
    
    # 变换车道
    transformed_lane = lane.copy()
    transformed_lane[:, 0] += tx
    transformed_lane[:, 1] += ty
    
    # 正确：极点也平移
    correct_transformed_pole = original_pole + np.array([tx, ty])
    
    # 错误：极点不变
    wrong_pole = original_pole.copy()
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('仿射变换（平移）时的极点参考系问题演示', fontsize=16, fontweight='bold')
    
    # 1. 原始图像
    draw_coordinate_system(
        axes[0], (0, 0), img_w, img_h,
        '1. 原始图像',
        lanes=[lane], pole=original_pole, show_polar=True,
        lane_color='blue', pole_color='red'
    )
    
    # 添加平移向量说明
    axes[0].annotate('', xy=(400, 180), xytext=(320, 160),
                    arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    axes[0].text(360, 165, f'平移\n({tx}, {ty})', fontsize=10, color='purple',
                ha='center', bbox=dict(boxstyle='round', facecolor='lavender'))
    
    # 2. 错误：只变换车道
    draw_coordinate_system(
        axes[1], (0, 0), img_w, img_h,
        '2. 变换后（错误：极点未平移）',
        lanes=[transformed_lane], pole=wrong_pole, show_polar=True,
        lane_color='orange', pole_color='red'
    )
    axes[1].text(img_w/2, img_h - 20, '❌ 极点未平移！', 
                fontsize=12, color='red', fontweight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor='yellow'))
    
    # 3. 正确：车道和极点都变换
    draw_coordinate_system(
        axes[2], (0, 0), img_w, img_h,
        '3. 变换后（正确：极点同步平移）',
        lanes=[transformed_lane], pole=correct_transformed_pole, show_polar=True,
        lane_color='green', pole_color='darkgreen'
    )
    axes[2].text(img_w/2, img_h - 20, '✓ 极点正确平移！', 
                fontsize=12, color='green', fontweight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = '/home/runner/work/PolarRCNN/PolarRCNN'
    output_path = os.path.join(output_dir, 'affine_transform_issue_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 演示图已保存到: {output_path}")
    plt.close()


def demonstrate_solution_architecture():
    """演示解决方案的架构"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.axis('off')
    
    # 标题
    title_text = "极点参考系修复方案 - 实现架构"
    ax.text(0.5, 0.95, title_text, transform=ax.transAxes,
           fontsize=18, fontweight='bold', ha='center')
    
    # 流程图内容
    flowchart = """
    ┌─────────────────────────────────────────────────────────────┐
    │                   数据增强前准备                              │
    │                                                               │
    │  1. 读取原始图像和车道标注                                     │
    │     img, lanes = get_sample(index)                           │
    │                                                               │
    │  2. 准备关键点列表（关键改动）                                 │
    │     keypoints = [lane_points..., center_point]               │
    │     ├─ 车道点：所有车道的所有点                                │
    │     └─ 极点：[center_w, center_h] ← 添加到末尾                │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                  Albumentations数据增强                       │
    │                                                               │
    │  content = train_augments(                                   │
    │      image=img,                                              │
    │      keypoints=keypoints  # 包含车道点和极点                  │
    │  )                                                            │
    │                                                               │
    │  自动变换：                                                    │
    │  • 图像 → 变换后的图像                                         │
    │  • 所有关键点 → 变换后的关键点                                 │
    │    （车道点和极点使用相同的变换矩阵）                           │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                  提取变换后的数据                              │
    │                                                               │
    │  img = content['image']                                      │
    │  keypoints = content['keypoints']                            │
    │                                                               │
    │  # 分离车道点和极点（关键步骤）                                │
    │  transformed_center = keypoints[-1]  # 最后一个是极点          │
    │  lane_keypoints = keypoints[:-1]     # 其余是车道点           │
    │                                                               │
    │  # 重建车道列表                                                │
    │  lanes = rebuild_lanes(lane_keypoints)                       │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │               生成Ground Truth（关键改动）                     │
    │                                                               │
    │  cls_gt, centerness_gt, theta_gt, r_gt =                     │
    │      generate_afpl_ground_truth(                             │
    │          lanes,                                              │
    │          transformed_center  # 使用变换后的极点！              │
    │      )                                                        │
    │                                                               │
    │  极坐标计算：                                                  │
    │  center_w_feat = transformed_center[0] / downsample          │
    │  center_h_feat = transformed_center[1] / downsample          │
    │                                                               │
    │  dx = x_coords - center_w_feat                               │
    │  dy = center_h_feat - y_coords                               │
    │  theta = arctan2(dy, dx)  # ✓ 基于正确的极点                 │
    │  r = sqrt(dx² + dy²)      # ✓ 基于正确的极点                 │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                    返回训练样本                               │
    │                                                               │
    │  return {                                                    │
    │      'img': img_tensor,                                      │
    │      'cls_gt': cls_gt,                                       │
    │      'centerness_gt': centerness_gt,                         │
    │      'theta_gt': theta_gt,  # ✓ 与增强后图像一致              │
    │      'r_gt': r_gt            # ✓ 与增强后图像一致             │
    │  }                                                            │
    └─────────────────────────────────────────────────────────────┘
    
    
    关键优势：
    ═══════════════════════════════════════════════════════════════
    
    ✓ 几何一致性：图像、车道点、极点使用相同变换矩阵
    ✓ 自动化：albumentations自动处理所有关键点变换
    ✓ 通用性：支持所有几何变换（翻转、旋转、仿射等）
    ✓ 简单性：只需在augment()中添加/提取极点
    ✓ 兼容性：不影响现有代码结构
    
    
    修改的文件：
    ═══════════════════════════════════════════════════════════════
    
    1. Dataset/afpl_base_dataset.py
       • augment() 方法：添加极点到关键点
       • generate_afpl_ground_truth()：使用transformed_center
    
    2. Dataset/base_dataset.py
       • augment() 方法：添加极点到关键点
       • fit_lane()：使用transformed_center
       • get_polar_map()：使用transformed_center
       • img2cartesian_with_center()：新增辅助方法
    """
    
    ax.text(0.05, 0.88, flowchart, transform=ax.transAxes,
           fontsize=9.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='blue', linewidth=2))
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = '/home/runner/work/PolarRCNN/PolarRCNN'
    output_path = os.path.join(output_dir, 'solution_architecture.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 架构图已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("="*60)
    print("极点参考系问题演示")
    print("="*60)
    print()
    
    print("正在生成演示图...")
    print()
    
    # 1. 水平翻转问题演示
    print("1. 生成水平翻转问题演示图...")
    demonstrate_horizontal_flip_issue()
    print()
    
    # 2. 仿射变换问题演示
    print("2. 生成仿射变换问题演示图...")
    demonstrate_affine_transform_issue()
    print()
    
    # 3. 解决方案架构
    print("3. 生成解决方案架构图...")
    demonstrate_solution_architecture()
    print()
    
    print("="*60)
    print("✓ 所有演示图生成完成！")
    print("="*60)
    print()
    print("生成的文件：")
    print("  1. pole_reference_issue_demo.png - 水平翻转问题演示")
    print("  2. affine_transform_issue_demo.png - 仿射变换问题演示")
    print("  3. solution_architecture.png - 解决方案架构")
    print()


if __name__ == '__main__':
    main()
