_base_ = [
    'final_cascade_rcnn_r50.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
evaluation = dict(
    classwise=True
)