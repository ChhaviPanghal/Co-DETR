CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}


# configs/co_dino/co_dino_5scale_lsj_swin_large_16e_o365tolvis.py
# C:\Users\lakshay\Desktop\detection\Co-DETR\models\co_dino_5scale_lsj_swin_large_16e_o365tolvis.pth
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch1.11.0/index.html
# pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
# sh tools/dist_test.sh  projects/configs/co_dino/co_dino_5scale_lsj_swin_large_16e_o365tolvis.py models/co_dino_5scale_lsj_swin_large_16e_o365tolvis.pth 1 --eval bbox
# sh tools/slurm_test.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint --eval bbox

python tools/test.py projects/configs/co_dino/co_dino_5scale_r50_1x_lvis.py train/latest.pth --eval bbox --work-dir test --out results/bboxes.pkl