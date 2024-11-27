# nohup python train.py --dataset /opt/data/private/wzf/u_shapley/BUSI_all --arch UNet --name shapley30_400_unet_shuffle --img_ext .png --mask_ext _mask.png --lr 0.001 --epochs 400 --b 8 --input_w 256 --input_h 256 --pretrain --pretrain-path /opt/data/private/wzf/u_shapley/pruning_segmentation/models/pretrain_400_unet_shuffle/model.pth --pruning --strategy shapley > shapley30_400_unet_shuffle.txt 2>&1 &

nohup python train.py --dataset /opt/data/private/wzf/u_shapley/dsb-unet++/dsb2018_96 --arch NestedUNet --name shapley30_400_unet_shuffle --img_ext .png --mask_ext _mask.png --lr 0.001 --epochs 100 --b 8 --input_w 256 --input_h 256 --pretrain --pretrain-path /opt/data/private/wzf/u_shapley/pruning_segmentation/models/pretrain_400_unet_shuffle/model.pth --pruning --strategy shapley > shapley30_400_unet_shuffle.txt 2>&1 &