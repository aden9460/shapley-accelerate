import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import autograd

import numpy as np

import sys, os
sys.path.append("/opt/data/private/wzf/u_shapley/shapley_accelerated/exp_adult")
sys.path.append("/opt/data/private/wzf/u_shapley/shapley_accelerated/adult-dataset")
# sys.path.append("/opt/data/private/wzf/u_shapley/Torch-Pruning/torch_pruning")

import tqdm
from adult_model import mlp, Model_for_shap
from train_adult import load_data, save_checkpoint
from captum.attr import *
import shap
from shap_utils import eff_shap
from eff_shap_utils import Efficient_shap_dnn
import argparse
import time
import torch_pruning as tp
from torch_pruning import function

class Shapley_accelereated():
    def __init__(self,model,data_generator):
        self.model = model
        self.data_generator = data_generator
        self.layer_outputs = {}
        self.reference = None
        self.target_module = None
        self.error_matrix = None
        self.whole_layer = None
    def run(self,layer):
        self.whole_layer = layer

        self.error_matrix = self.calc_cross_contribution(self.whole_layer)
        shapley_value_buf = self.efficient_shap(self.model)
        
        return shapley_value_buf


    def hook_fn(self, module, input, output):
        self.layer_outputs[module] = output


    def get_second_order_grad(self, layer, device=None):

        # x = torch.FloatTensor(x).to(device)
        # print(x.dtype)

        with torch.set_grad_enabled(True):

            # if x.nelement() < 2:
            #     return np.array([])

            # x.requires_grad = True


            hook = layer.register_forward_hook(self.hook_fn)


            for idx, (x,labels) in enumerate(self.data_generator):


                y = self.model(x)
                loss_fn =  nn.CrossEntropyLoss()
                loss = loss_fn(y, labels)
                # target_neuron = self.layer_outputs[layer]
                target_neuron = layer.weight
                self.target_module = target_neuron
                # y = y.sum()
                # grad2 = torch.autograd.functional.hessian(y, target_neuron)
                grads = autograd.grad(loss, target_neuron, create_graph=True,retain_graph=True)[0].squeeze()

                # grads = torch.sum(grads)

                grad_list = []
                for j, grad in enumerate(grads):
                    grad2 = autograd.grad(grad.sum(), target_neuron, create_graph=True,retain_graph=True,allow_unused=False)[0]
                    grad_list.append(grad2)
                    # grad2 = autograd.grad(grad, target_neuron)[0].squeeze()

                # for i in range(len(target_neuron.squeeze())):
                # grad2 = autograd.grad(grads, target_neuron, create_graph=True,retain_graph=True,allow_unused=False)[0]
                    # grad2_y = torch.autograd.grad(grad_y[i], target_neuron, create_graph=True)[0]
                    # grad_list.append(grad2)
                

               
                    # grad_list.append(grad2)
                # grad_matrix = torch.stack(grad2).squeeze()    
                # self.reference = torch.full((1,len(target_neuron.squeeze())),target_neuron.mean().item()).squeeze()
                self.reference = torch.mean(target_neuron,dim=1)
                hook.remove()
                return torch.abs(grad2) #先暂时用1张
        
def get_second_order_weight_grad(model, x, device=None):
    with torch.set_grad_enabled(True):
        if x.nelement() < 2:
            return np.array([])

        x.requires_grad = True

        y = model(x)
        grads = autograd.grad(y, model.parameters(), create_graph=True)

        grad_list = []
        for grad in grads:
            grad2_list = []
            for g in grad.view(-1):
                grad2 = autograd.grad(g, model.parameters(), retain_graph=True)
                grad2_list.append(torch.cat([g2.view(-1) for g2 in grad2]))
            grad_list.append(torch.stack(grad2_list))

        grad_matrix = torch.cat(grad_list, dim=0)
        return torch.abs(grad_matrix)



    @torch.no_grad()
    def distance_estimation_s(self):

        # reference_sparse = torch.cat([x[-1] for x in cate_attrib_book], dim=0)
        # dense_feat_num = dense_feat_index.shape[0]
        # reference = torch.cat((reference[0:dense_feat_num], reference_sparse), dim=0)
    
 

        distance_vector = torch.abs(self.target_module - self.reference).squeeze(dim=0)

        # distance_i, distance_j = torch.meshgrid(distance_vector, distance_vector)
        # distance_matrix = distance_i * distance_j

        return distance_vector


    # 计算特征之间的交互贡献
    def calc_cross_contribution(self,layer):
        feat_num = layer.out_features # 算当前层的特征数
        interaction_matrix = torch.zeros((feat_num, feat_num))

        # data_iter = iter(self.data_generator)
        # images, labels = data_iter.next() 
        
        # # 将梯度展平成一维向量
        # input_grad_flat = model.view(-1)
        error_matrix = []
        # for i in range(feat_num):
        #     for j in range(feat_num):
                # 计算每两个特征的交叉贡献（这里用点积表示）    =
        grad_matrix_1 = self.get_second_order_grad(layer)
        distance_matrix = self.distance_estimation_s()

        # interaction_matrix[i, j] = input_grad_flat[i] * input_grad_flat[j]  # 或者使用 torch.dot() 计算点积
        # interaction_matrix[i, j] = grad_matrix_1[i] * distance_matrix[j] 
        interaction_matrix = grad_matrix_1 * distance_matrix


        return interaction_matrix

    @torch.no_grad()
    def efficient_shap(self,model_for_shap):
        # model.eval()
        feat_num = self.target_module.shape[-1]

        shapley_value_buf = []
        shapley_rank_buf = []
        topK = torch.log2(torch.tensor(feat_num)).type(torch.long) - 1
        MSE_buf = []
        mAP_buf = []
        total_time = 0

        eff_shap_agent = Efficient_shap_dnn(model_for_shap, self.reference, topK,self.target_module,whole_layer=self.whole_layer,data_gen=self.data_generator)

        # for index, (x, y, z, sh_gt, rk_gt, error_matrix) in enumerate(data_loader):
            # x = x[0].unsqueeze(dim=0)
            # z = z[0].unsqueeze(dim=0)
            # shapley_gt = sh_gt
            # ranking_gt = rk_gt
        # for index, (x, z) in enumerate(self.target_module):
        z = self.target_module
        # feat_num = z.shape[-1]
        error_matrix = self.error_matrix.squeeze(dim=0).numpy()

        eff_shap_agent.feature_selection(error_matrix)

        # error_matrix = error_term_estimation(model_grad, x, reference, dense_feat_index, sparse_feat_index, cate_attrib_book).detach().numpy()

        # shapley_groundtruth = torch.tensor([[ 2.1027e-03,  2.5598e-04,  9.2007e-03,  1.3041e-04,  2.0145e-03,
        #   1.2062e-02,  8.6793e-02, -3.4935e-01, -6.9856e-02, -2.1126e-01,
        #   1.8807e-02,  5.5653e-02, -3.4564e-01]])

        # print(z)
        # print(shapley_gt)
        # print(shapley_groundtruth)

        attr_buf = []
        for idx in range(feat_num):
            t0 = time.time()
            # attr = eff_shap(model_for_shap, z, reference, error_matrix, topK)
            attr = eff_shap_agent.forward(z)
            # t1 = time.time()
            # print(index, t1-t0)
            total_time += time.time() - t0
            attr_buf.append(attr)

        attr_buf = torch.cat(attr_buf, dim=0)
        attr = attr_buf.mean(dim=0).reshape(1, -1)

        # ranking = attr.argsort(dim=1)
        shapley_value_buf.append(attr)
        # shapley_rank_buf.append(ranking)

            # rank_mAP = ((ranking == ranking_gt).sum(dim=1).type(torch.float)/ranking_gt.shape[-1]).mean(dim=0)
            # mAP_buf.append(rank_mAP)

            # MSE = torch.square(attr - shapley_gt).mean(dim=0)  # .sum(dim=1) #
            # MSE_sum = torch.square(attr - shapley_gt).sum(dim=1).mean(dim=0)  # #
            # MSE_buf.append(MSE_sum)
            # MMSE = torch.tensor(MSE_buf).mean(dim=0)
            # mAP = torch.tensor(mAP_buf).mean(dim=0)
            # mAP_std = np.array(mAP_buf).std(axis=0)

            # print("Index: {}, mAP: {}, MMSE {}".format(index, rank_mAP, MMSE))
            # print("Index: {}, Rank Precision: {}, mAP: {}, mAP std: {}".format(index, rank_mAP, mAP, mAP_std))

            # stop
            # if index == 100:
            #     break

        shapley_value_buf = torch.cat(shapley_value_buf, dim=0)
        # shapley_rank_buf = torch.cat(shapley_rank_buf, dim=0)

        # return shapley_value_buf, shapley_rank_buf, total_time
        return shapley_value_buf


if __name__ == '__main__': 
    # 加载 CIFAR-10 数据集
    transform = transforms.Compose([transforms.Resize(224),  # VGG 网络期望输入 224x224 的图像
                                    transforms.ToTensor()])

    # 加载训练集和测试集
    train_dataset = torchvision.datasets.CIFAR10(root='/opt/data/common/datasets/cifar10', train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

    # 使用预训练的 VGG16 模型
    model = vgg16(pretrained=False,num_classes=10)
    model.eval()  # 设置模型为评估模式

    # 获取 CIFAR-10 的一个样本
    data_iter = iter(train_loader)
    images, labels = data_iter.next()  # 获取一张图片和标签
    images.requires_grad = True  # 允许计算输入的梯度

    # # 前向传播
    # output = vgg_model(images)

    # # 计算输出关于输入的梯度
    # vgg_model.zero_grad()  # 清除梯度
    # output.backward(torch.ones_like(output))  # 反向传播

    # # 获取输入图像的梯度
    # input_grad = images.grad  # input_grad 现在是输入的梯度，表示每个像素对输出的贡献

    DG = tp.DependencyGraph()
    DG.build_dependency(model, images)

    
    # print(groups)
    # for g in groups:
    #    print(g)
    # 计算交叉贡献矩阵
    
    result = []
    attr = Shapley_accelereated(model,train_loader)
    # for idxs,dep in enumerate(groups):
    for group in DG.get_all_groups():
            # idxs.sort()
        for dep,idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            if isinstance(layer, nn.Linear):
                if layer.out_features==10:
                    continue

            # root_idxs = groups[i].root_idxs

            # if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                shapley_value = attr.run(layer)
                result.append(shapley_value)

    # interaction_matrix,reference = calc_cross_contribution(model,train_loader)

    # model_for_shap = Model_for_shap(model, train_loader)

    # # shapley_value, shapley_rank, total_time = efficient_shap(model_for_shap.forward_1, data_loader, reference,
    #                                                 #  dense_feat_index, sparse_feat_index, cate_attrib_book)
    # shapley_value = efficient_shap(model_for_shap.forward_1, train_loader,interaction_matrix)





# print("交叉贡献矩阵的形状：", interaction_matrix.shape)
