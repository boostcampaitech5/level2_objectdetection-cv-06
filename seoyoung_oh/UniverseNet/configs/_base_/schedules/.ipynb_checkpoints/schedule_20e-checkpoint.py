# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     # warmup='linear',
#     # warmup_iters=500,
#     # warmup_ratio=0.001,
#     # step=[16, 19]
#     gamma=0.5,
#     step=[12,19]
# )
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.001*optimizer['lr'],
    by_epoch=True
)
lr_config = dict(policy='fixed')
runner = dict(type='EpochBasedRunner', max_epochs=20)
