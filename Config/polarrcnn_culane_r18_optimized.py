cfg_name = 'polarrcnn_culane_r18_optimized'
############### import package ######################
import math
import cv2

############### dataset choise ######################
dataset =  'culane'
data_root = './dataset/CULane'

############### image parameter #########################
ori_img_h =  590
ori_img_w =  1640
cut_height =  270
img_h = 320
img_w = 800
center_h = 25
center_w = 386
max_lanes = 4

############## data augment ###############################
# Enhanced augmentations for night scene performance
train_augments = [
     dict(name='Resize', parameters=dict(height=img_h, width=img_w, interpolation=cv2.INTER_CUBIC, p=1.0)),
     dict(name='HorizontalFlip', parameters=dict(p=0.5)),
     # Enhanced brightness range for better night scene handling
     dict(name='RandomBrightnessContrast', parameters=dict(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.15), p=0.7)),
     # Increased HSV augmentation for varied lighting conditions
     dict(name='HueSaturationValue', parameters=dict(hue_shift_limit=(-10, 10), sat_shift_limit=(-15, 15), val_shift_limit=(-10, 10), p=0.75)),
     # Add CLAHE (Contrast Limited Adaptive Histogram Equalization) for night scenes
     dict(name='CLAHE', parameters=dict(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)),
     # Motion and Median blur for robustness
     dict(name='OneOf', transforms=[dict(name='MotionBlur', parameters=dict(blur_limit=(3, 5)), p=1.0),
                                    dict(name='MedianBlur', parameters=dict(blur_limit=(3, 5)), p=1.0)], p=0.2),
     # Add GaussNoise for night scene robustness
     dict(name='GaussNoise', parameters=dict(var_limit=(10.0, 50.0), p=0.25)),
     # Geometric augmentation
     dict(name='Affine', parameters=dict(translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)), rotate=(-9, 9), scale=(0.8, 1.2), interpolation=cv2.INTER_CUBIC, p=0.7)),
     dict(name='Resize', parameters=dict(height=img_h, width=img_w, interpolation=cv2.INTER_CUBIC, p=1.0)),
]

############### lane parameter #########################
num_offsets = 72
offset_stride = 4.507

######################network parameter#################################
#####backbone#####
backbone = 'resnet18'
pretrained = True

#####neck#####
neck = 'fpn'
fpn_in_channel = [128, 256, 512]
# Reduced neck dimension for lightweight model (64 -> 48)
neck_dim = 48
downsample_strides = [8, 16, 32]

#####rpn head#####
rpn_head = 'local_polar_head'
rpn_inchannel = neck_dim
polar_map_size = (4, 10)
num_training_priors = polar_map_size[0]*polar_map_size[1]
num_testing_priors = 20
angle_noise_p = 0.025
rho_noise_p = 0.25

#####roi head#####
roi_head = 'global_polar_head'
num_feat_samples = 36
# Reduced hidden dimension for lightweight model (192 -> 144)
fc_hidden_dim = 144
prior_feat_channels = 48
num_line_groups = 6
# Reduced GNN dimension for lightweight model (128 -> 96)
gnn_inter_dim = 96
iou_dim = 5
o2o_angle_thres = math.pi/6
o2o_rho_thres = 50

############## train parameter ###############################
batch_size = 40
# Increased epochs for better convergence (32 -> 36)
epoch_num = 36
random_seed = 3404

######################optimizer parameter#################################
lr = 6e-4
warmup_iter = 800

######################loss parameter######################################
rpn_loss = 'polarmap_loss'
roi_loss = 'tribranch_loss'

#####cost function#####
reg_cost_weight = 6
reg_cost_weight_o2o = 6
cls_cost_weight = 1
angle_prior_thres = math.pi/5
rho_prior_thres = 80
cost_iou_width = 30
ota_iou_width = 7.5

#####loss function #####
g_weight = 1
iou_loss_weight = 2
cls_loss_weight = 0.33
# Adjusted for better balance on difficult scenes
cls_loss_alpha = 0.45
cls_loss_alpha_o2o = 0.28
# Reduced rank loss weight for better generalization (0.7 -> 0.5)
rank_loss_weight = 0.5
end_loss_weight = 0.03
aux_loss_weight = 0.2
polarmap_loss_weight = 5
loss_iou_width = 7.5

######################postprocess parameter######################################
nms_thres = 50
# Lowered confidence threshold for better recall on night scenes (0.48 -> 0.45)
conf_thres = 0.45
conf_thres_o2o = conf_thres
# Adjusted NMS-free threshold for consistency (0.46 -> 0.43)
conf_thres_nmsfree = 0.43
is_nmsfree = True
