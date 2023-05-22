import cv2
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

image_scale = [(256, 256), (512, 512), (768, 768), (1024, 1024)]

albu_train_transforms = [ 
#    dict(type='HorizontalFlip',p=0.5),
#    dict(type='VerticalFlip',p=0.5),
#    dict(
#        type='OneOf',
#        transforms=[
#            dict(
#                type='ShiftScaleRotate',
#                shift_limit=0.07,
#                scale_limit=0.0,
#                rotate_limit=0.0,
#                border_mode=cv2.BORDER_CONSTANT,
#                p=1),
#            dict(
#                type='ShiftScaleRotate',
#                shift_limit=0.0,
#                scale_limit=0.2,
#                rotate_limit=0.0,
#                border_mode=cv2.BORDER_CONSTANT,
#                p=1),
#            dict(
#                type='ShiftScaleRotate',
#                shift_limit=0.0,
#                scale_limit=0.0,
#                rotate_limit=30,
#                border_mode=cv2.BORDER_CONSTANT,
#                p=1),
#        ],
#        p=0.1),
#    dict(
#        type='Cutout',
#        num_holes=8, 
#        max_h_size=48, 
#        max_w_size=48, 
#        p=0.5),
#    dict(
#       type='OneOf',
#        transforms=[
#            dict(type='Blur', blur_limit=3, p=1.0),
#            dict(type='MedianBlur', blur_limit=3, p=1.0)
#        ],
#        p=0.1),

]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale = image_scale, multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
dict(
        type='MultiScaleFlipAug',
        img_scale=image_scale,
        flip=False,
        transforms=[
            dict(type='Resize', 
                 img_scale=image_scale,
                 multiscale_mode='value',
                 keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= data_root + 'train_1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file= data_root + 'val_1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')