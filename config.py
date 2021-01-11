seed = 42
gpus = [0]
batch_size = 32
epochs = 30
num_workers = 8

train_dataset_len = 91756 // batch_size
height = 320
width = 544
classes = ['bare_head', 'helmet', 'ear_protection', 'welding_mask',
           'bare_chest', 'high_visibility_vest', 'person']
num_classes = 7
background_color = (0, 0, 0)

trainer_cfg = dict(
    gpus=gpus,
    max_epochs=epochs,
    callbacks=[
        dict(type='LearningRateMonitor', logging_interval='step'),
        dict(type='ModelCheckpoint', save_top_k=5, verbose=True, mode='max',
             monitor='mAP_0.95', dirpath='./results/',
             filename='{epoch:02d}_{rank_1:.2f}')
    ],
    benchmark=True,
    deterministic=True,
    terminate_on_nan=True,
    distributed_backend='ddp',
    precision=16,
    sync_batchnorm=True,
)
wandb_cfg = dict(
    name=f'{__file__.split("/")[-1].replace(".py", "")}_{height}_{width}_{batch_size}_ep{epochs}',
    project='equipment_detector'
)

backbone_cfg = dict(
    type='ResNet', depth=34
)

loss_head_cfg = dict(
    type='TTFHead',
    num_classes=num_classes
)

metric_cfgs = [
    dict(type='mAP',
         labels_ids=dict(zip(range(len(classes)), classes)),
         iou_treshold=0.5,
         name='mAP_0.5'),
    dict(type='mAP',
         labels_ids=dict(zip(range(len(classes)), classes)),
         iou_treshold=0.75,
         name='mAP_0.75'),
    dict(type='mAP',
         labels_ids=dict(zip(range(len(classes)), classes)),
         iou_treshold=0.95,
         name='mAP_0.95'),
]

train_transforms_cfg = dict(
    type='Compose', transforms=[
        dict(type='Resize', width=width, height=height),
        dict(type='RandomBrightnessContrast', brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        dict(type='RGBShift', r_shift_limit=(10, 20), g_shift_limit=(10, 20), b_shift_limit=(10, 20), p=0.7),
        dict(type='OneOf', transforms=[
            dict(type='MotionBlur', p=1.),
            dict(type='Blur', blur_limit=3, p=1.),
            dict(type='MedianBlur', blur_limit=3, p=1.)
        ], p=0.2),
        dict(type='HueSaturationValue', p=0.3),
        dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
        dict(type='ToTensorV2')
    ])

val_transforms_cfg = dict(
    type='Compose', transforms=[
        dict(type='Resize', width=width, height=height),
        dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
        dict(type='ToTensorV2')
    ])

train_dataset_cfg = dict(
    type='VWPPEDataset',
    path_to_dir='/home/ivan/MLTasks/Datasets/ObjectDetection/PersonEquipmentTask',
    is_train=True,
    debug=False
)

val_dataset_cfg = dict(
    type='VWPPEDataset',
    path_to_dir='/home/ivan/MLTasks/Datasets/ObjectDetection/PersonEquipmentTask',
    is_train=False,
    debug=False
)

train_dataloader_cfg = dict(
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

val_dataloader_cfg = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

optimizer_cfg = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9
)
scheduler_cfg = dict(
    type='CyclicLR',
    base_lr=1e-6 * len(gpus),
    max_lr=1e-4 * len(gpus),
    step_size_up=int(train_dataset_len // batch_size * (epochs * 0.1)),
    mode='triangular2',
    cycle_momentum=False,
)
scheduler_update_params = dict(
    interval='step',
    frequency=1
)

module_cfg = dict(
    type='LightningEquipmentDetNet',
    backbone_cfg=backbone_cfg,
    loss_head_cfg=loss_head_cfg,
    metric_cfgs=metric_cfgs,
    train_transforms_cfg=train_transforms_cfg,
    val_transforms_cfg=val_transforms_cfg,
    train_dataset_cfg=train_dataset_cfg,
    val_dataset_cfg=val_dataset_cfg,
    train_dataloader_cfg=train_dataloader_cfg,
    val_dataloader_cfg=val_dataloader_cfg,
    optimizer_cfg=optimizer_cfg,
    scheduler_cfg=scheduler_cfg,
    scheduler_update_params=scheduler_update_params
)