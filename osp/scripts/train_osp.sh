

CUDA_VISIBLE_DEVICES=0 python train.py   --loss1 1.5 --loss2 0.5 --trainer OSPCoOp --shots 16 --seed 1  --config-file "configs/trainers/OSPCoOp/vit_b16_ep50.yaml"  --output_dir "./results_all//16shots_3.5_96"  --id_aug_type 'mix' --id_aug_thre 3.5 --num_aug_versions 96
