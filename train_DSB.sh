# python train.py --dataset /opt/data/private/wzf/u_shapley/dsb-unet++/dsb2018_96 --arch NestedUNet

python train.py --dataset /opt/data/private/wzf/u_shapley/dsb-unet++/dsb2018_96/ --arch UNet --name dsb_unet_pretrain400 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 
python train.py --dataset /opt/data/private/wzf/u_shapley/dsb-unet++/dsb2018_96/ --arch NestedUNet --name dsb_NestedUNet_pretrain40 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 

python train.py --dataset /opt/data/private/wzf/u_shapley/dsb-unet++/dsb2018_96/ --arch UNet --name dsb_unet_pretrain400_shapley0.3 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_unet_pretrain400/model.pth --pruning --strategy shapley > dsb_unet_pretrain400_shapley0.3.txt
python train.py --dataset /opt/data/private/wzf/u_shapley/dsb-unet++/dsb2018_96/ --arch NestedUNet --name dsb_NestedUNet_pretrain40_shapley0.3 --img_ext .png --mask_ext .png --lr 0.001 --epochs 400 --b 8 --input_w 96 --input_h 96 --pretrain --pretrain-path models/dsb_NestedUNet_pretrain40/model.pth --pruning --strategy shapley > dsb_NestedUNet_pretrain40_shapley0.3.txt
