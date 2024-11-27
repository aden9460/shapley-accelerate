#!/bin/bash

python train.py --dataset /opt/data/private/datasets/isic/ --arch UNet --name new_isic_unet_rank_flops_50 --img_ext .jpg --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 512 --input_h 512 --pretrain --pretrain-path models/isic_unet/model.pth --pruning --strategy rank > new_isic_unet_rank_flops_50.txt
python train.py --dataset /opt/data/private/datasets/isic/ --arch NestedUNet --name new_isic_unet++_rank_flops_50 --img_ext .jpg --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 512 --input_h 512 --pretrain --pretrain-path models/isic_unet++/model.pth --pruning --strategy rank > new_isic_unet++_rank_flops_50.txt

python train.py --dataset /opt/data/private/datasets/isic/ --arch UNet --name new_isic_unet_shapley_flops_50 --img_ext .jpg --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 512 --input_h 512 --pretrain --pretrain-path models/isic_unet/model.pth --pruning --strategy shapley > new_isic_unet_shapley_flops_50.txt
python train.py --dataset /opt/data/private/datasets/isic/ --arch NestedUNet --name new_isic_unet++_shapley_flops_50 --img_ext .jpg --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 512 --input_h 512 --pretrain --pretrain-path models/isic_unet++/model.pth --pruning --strategy shapley > new_isic_unet++_shapley_flops_50.txt
