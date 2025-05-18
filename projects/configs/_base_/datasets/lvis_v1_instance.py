# dataset settings
_base_ = 'coco_instance.py'
dataset_type = 'LVISV1Dataset'
# data_root = 'data/lvis_v1/'
data_root = 'Data/Det/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root+'train/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root+'val/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root+'test/'))
evaluation = dict(metric=['bbox'])
