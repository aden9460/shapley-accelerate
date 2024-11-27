#!/bin/bash

#python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch UNet --name new_dsb_unet_random_flops_50 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet/model.pth --pruning --strategy random > new_dsb_unet_random_flops_50.txt
#python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch NestedUNet --name new_dsb_unet++_random_flops_50 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet++/model.pth --pruning --strategy random > new_dsb_unet++_random_flops_50.txt
#
#python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch UNet --name new_dsb_unet_l1_flops_50 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet/model.pth --pruning --strategy l1 > new_dsb_unet_l1_flops_50.txt
#python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch NestedUNet --name new_dsb_unet++_l1_flops_50 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet++/model.pth --pruning --strategy l1 > new_dsb_unet++_l1_flops_50.txt
#
#python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch UNet --name new_dsb_unet_rank_flops_50 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet/model.pth --pruning --strategy rank > new_dsb_unet_rank_flops_50.txt
#python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch NestedUNet --name new_dsb_unet++_rank_flops_50 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet++/model.pth --pruning --strategy rank > new_dsb_unet++_rank_flops_50.txt
#
#python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch UNet --name new_dsb_unet_shapley_flops_50 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet/model.pth --pruning --strategy shapley > new_dsb_unet_shapley_flops_50.txt
#python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch NestedUNet --name new_dsb_unet++_shapley_flops_50 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet++/model.pth --pruning --strategy shapley > new_dsb_unet++_shapley_flops_50.txt

for s in $(seq 0.1 0.1 0.9); do
    # 内层循环，四个元素列表
    for t in "random" "l1" "rank" "shapley"; do
        # 执行Python命令
        python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch UNet --name "dsb_unet_${t}_flops_50_${s}" --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet/model.pth --pruning --strategy "$t" --sample-ratio "$s" > "dsb_unet_${t}_flops_50_sample${s}.txt"
        python train.py --dataset /opt/data/private/datasets/dsb-unet++/dsb2018_96/ --arch NestedUNet --name "dsb_unet++_${t}_flops_50_${s}" --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet++/model.pth --pruning --strategy "$t" --sample-ratio "$s" > "dsb_unet++_${t}_flops_50_sample${s}.txt"
    done
done