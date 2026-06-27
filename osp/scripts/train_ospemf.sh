CUDA_VISIBLE_DEVICES=0   python train.py   --loss1 0.5 --loss2 0.5 --trainer OSPCoOp_EMF --shots 16 --seed 1 --output_dir ./runs_all/loss_ab_new/OSPCoOp_EMF_16shots_k200_c96_3.5_64_0.5_0.5 --start_eval_epoch 5 --crop_num_per_img 96  --topk 200  --id_aug_thre 3.5 --num_aug_versions 64

