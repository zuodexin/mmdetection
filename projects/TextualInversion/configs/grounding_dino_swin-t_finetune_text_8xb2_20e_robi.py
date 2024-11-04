_base_ = ['../../../configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py']

load_from = "weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth"

custom_imports = dict(imports=['projects.TextualInversion.datasets'])

_base_.dataset_type = 'ROBIDataset'
_base_.data_root = 'data/robi/'

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.),
            'neck': dict(lr_mult=0.),
            'encoder': dict(lr_mult=0.),
            'decoder': dict(lr_mult=0.),
            'positional_encoding': dict(lr_mult=0.),
            'bbox_head': dict(lr_mult=0.),
            # 'language_model': dict(lr_mult=0),
        }))


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=_base_.train_pipeline,
        backend_args=_base_.backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=_base_.test_pipeline,
        backend_args=_base_.backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=_base_.test_pipeline,
        backend_args=_base_.backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=_base_.data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    backend_args=_base_.backend_args)


test_evaluator = dict(
    type='CocoMetric',
    ann_file=_base_.data_root + 'annotations/instances_test.json',
    metric='bbox',
    format_only=False,
    backend_args=_base_.backend_args,
    outfile_prefix='./work_dirs/robi_results/textual_inversion'
    )