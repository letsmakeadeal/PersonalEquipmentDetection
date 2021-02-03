seed = 42
gpus = [0]
batch_size = 32
epochs = 30
num_workers = 8

train_dataset_len = 81859 // batch_size
height = 640
width = 1088
classes = ['bare_head', 'helmet', 'ear_protection', 'welding_mask',
           'bare_chest', 'high_visibility_vest', 'person']
num_classes = 7
background_color = (0, 0, 0)
divider = 32

trainer_cfg = dict(
    gpus=gpus,
    max_epochs=epochs,
    callbacks=[
        dict(type='LearningRateMonitor', logging_interval='step'),
        dict(type='ModelCheckpoint', save_top_k=2, verbose=True, mode='max',
             monitor='mAP_05', dirpath='./results/',
             filename='virtual_{epoch:02d}_{mAP_05:.4f}')
    ],
    benchmark=True,
    deterministic=True,
    terminate_on_nan=True,
    distributed_backend='ddp',
    precision=16,
    sync_batchnorm=True
)
wandb_cfg = dict(
    name=f'{__file__.split("/")[-1].replace(".py", "")}_{height}_{width}_{batch_size}_ep{epochs}',
    project='equipment_detector'
)

backbone_cfg = dict(
    type='ResNet',
    depth=34,
    frozen_stages=3
)

loss_head_cfg = dict(
    type='TTFHead',
    num_classes=num_classes
)

metric_cfgs = [
    dict(type='mAP',
         labels_ids=dict(zip(range(len(classes)), classes)),
         iou_treshold=0.5,
         name='mAP_05'),
    dict(type='mAP',
         labels_ids=dict(zip(range(len(classes)), classes)),
         iou_treshold=0.75,
         name='mAP_075'),
    dict(type='mAP',
         labels_ids=dict(zip(range(len(classes)), classes)),
         iou_treshold=0.95,
         name='mAP_095'),
]

train_transforms_cfg = dict(
    type='Compose', transforms=[
        dict(type='LongestMaxSize', max_size=max(width, height)),
        dict(type='PadIfNeeded', min_width=(width // divider) * divider, min_height=(height // divider) * divider,
             value=(0, 0, 0), border_mode=0),
        dict(type='CenterCrop', width=(width // divider) * divider, height=(height // divider) * divider),
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
        dict(type='LongestMaxSize', max_size=max(width, height)),
        dict(type='PadIfNeeded', min_width=(width // divider) * divider, min_height=(height // divider) * divider,
             value=(0, 0, 0), border_mode=0),
        dict(type='CenterCrop', width=(width // divider) * divider, height=(height // divider) * divider),
        dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
        dict(type='ToTensorV2')
    ])

train_dataset_cfg = dict(
    type='VWPPEDataset',
    mode='virtual_train',
    path_to_dir='/home/ivan/MLTasks/Datasets/ObjectDetection/PersonEquipmentTask',
    debug=False
)

val_dataset_cfg = dict(
    type='VWPPEDataset',
    mode='virtual_val',
    path_to_dir='/home/ivan/MLTasks/Datasets/ObjectDetection/PersonEquipmentTask',
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
    max_lr=1e-3 * len(gpus),
    step_size_up=int(train_dataset_len * epochs // (2 * len(gpus))),
    cycle_momentum=False,
)
scheduler_update_params = dict(
    interval='step',
    frequency=1
)

module_cfg = dict(
    type='LightningEquipmentDetNet',
    load_from_checkpoint=None,
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