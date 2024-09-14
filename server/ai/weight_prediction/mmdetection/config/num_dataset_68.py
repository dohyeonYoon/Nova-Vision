_base_ = '../../mmdetection/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('chicken')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/scratch/dohyeon/BRODY/src/method_override/dataset/dataset_68/annotations/train.json',
        img_prefix='/scratch/dohyeon/BRODY/src/method_override/dataset/dataset_68/train'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/scratch/dohyeon/BRODY/src/method_override/dataset/dataset_68/annotations/val.json',
        img_prefix='/scratch/dohyeon/BRODY/src/method_override/dataset/dataset_68/val'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/scratch/dohyeon/BRODY/src/method_override/dataset/dataset_68/annotations/val.json',
        img_prefix='/scratch/dohyeon/BRODY/src/method_override/dataset/dataset_68/val'))

# 2. model settings.
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))