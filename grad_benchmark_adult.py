import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import autograd

import numpy as np

import sys, os
sys.path.append("./adult-dataset")
sys.path.append("/opt/data/private/wzf/u_shapley/shapley_accelerated/adult-dataset")
import tqdm
from adult_model import mlp, Model_for_shap
from train_adult import load_data, save_checkpoint
from captum.attr import *
import shap
from shap_utils import eff_shap
import argparse

# sys.path.append("../DeepCTR-Torch/deepctr_torch/models")


def get_second_order_grad(model, x, device=None):

    # x = torch.FloatTensor(x).to(device)
    # print(x.dtype)

    with torch.set_grad_enabled(True):

        if x.nelement() < 2:
            return np.array([])

        x.requires_grad = True

        y = model(x)
        grads = autograd.grad(y, x, create_graph=True)[0].squeeze()#形状为[1, 107]

        grad_list = []
        for j, grad in enumerate(grads):
            grad2 = autograd.grad(grad, x, retain_graph = True)[0].squeeze()#形状为
            grad_list.append(grad2)

        grad_matrix = torch.stack(grad_list)#形状为[107, 107]
        return torch.abs(grad_matrix)

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
def distance_estimation(x, reference, dense_feat_index, cate_attrib_book):

    reference_sparse = torch.cat([x[-1] for x in cate_attrib_book], dim=0)
    dense_feat_num = dense_feat_index.shape[0]
    reference = torch.cat((reference[0:dense_feat_num], reference_sparse), dim=0)  #平均值

    distance_vector = torch.abs(x - reference.unsqueeze(dim=0)).squeeze(dim=0) #当前值剪去平均值

    distance_i, distance_j = torch.meshgrid(distance_vector, distance_vector)
    distance_matrix = distance_i * distance_j                                  #特征i和特征j的距离乘积

    return distance_matrix

# 数据共有107种特征，但是一共只有13种特征类型，我们要判断这13种特征类型对于结果的贡献，就要将107种特征的稀疏变量用one-hot编码，之后取其贡献最大值作为该种特征类型的贡献
@torch.no_grad()
def error_term_estimation(mlp, x, reference, dense_feat_index, sparse_feat_index, cate_attrib_book):
    interaction_scores = {}

    # if grad_gpu == -1:
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda:" + str(grad_gpu))
    #
    # mlp = mlp.to(device)

    device = torch.device("cpu")

    grad_matrix_1 = get_second_order_grad(mlp, x, device)
    distance_matrix = distance_estimation(x, reference, dense_feat_index, cate_attrib_book)

    # reference_sparse = torch.cat([x[-1] for x in cate_attrib_book], dim=0)
    # dense_feat_num = dense_feat_index.shape[0]
    # reference = torch.cat((reference[0:dense_feat_num], reference_sparse), dim=0)

    error_matrix = grad_matrix_1 * distance_matrix   #【107，107】

    dense_feat_num = dense_feat_index.shape[0]
    sparse_feat_num = sparse_feat_index.shape[0]

    error_matrix_comb = torch.zeros((dense_feat_num + sparse_feat_num,
                                    dense_feat_num + sparse_feat_num))

    error_matrix_comb[0:dense_feat_num, 0:dense_feat_num] = error_matrix[0:dense_feat_num, 0:dense_feat_num]

    block_len = torch.tensor([1 for x in dense_feat_index] + [x.shape[-1] for x in cate_attrib_book]).type(torch.int)  #获取每一类特征的长度 稠密的长度为1 分类特征的长度由其独热编码决定
    for feature_index_i in range(dense_feat_num + sparse_feat_num):

        for feature_index_j in range(dense_feat_num + sparse_feat_num):

            if feature_index_i == feature_index_j:
                error_matrix_comb[feature_index_i, feature_index_j] = -1
                continue

            elif feature_index_i < dense_feat_num and feature_index_j < dense_feat_num:
                continue

            feature_i_dim = block_len[feature_index_i]
            feature_j_dim = block_len[feature_index_j]

            index_strt_i = block_len[0:feature_index_i].sum()
            index_strt_j = block_len[0:feature_index_j].sum()

            error_matrix_comb[feature_index_i, feature_index_j] = torch.max(error_matrix[index_strt_i:index_strt_i+feature_i_dim,
                                                                           index_strt_j:index_strt_j+feature_j_dim])#在同一类型中选择最大的
    # print(error_matrix_comb)

    return error_matrix_comb


@torch.no_grad()
def grad_estimation(model_grad, data_loader, reference, dense_feat_index, sparse_feat_index, cate_attrib_book): # , shapley_value_gt, shapley_ranking_gt):
    # model.eval()

    feat_num = len(dense_feat_index) + len(sparse_feat_index)
    error_matrix_buf = torch.zeros((len(data_loader), feat_num, feat_num))
    for index, (x, _, _, _, _) in enumerate(data_loader):
        x = x[0].unsqueeze(dim=0)#输入形状为[1, 107]

        error_matrix = error_term_estimation(model_grad, x, reference, dense_feat_index, sparse_feat_index, cate_attrib_book).detach()
        error_matrix_buf[index] = error_matrix
        print(index)

        # if index == 10:
        #     break

    return error_matrix_buf



parser = argparse.ArgumentParser()
parser.add_argument("--softmax", action='store_true', help="softmax model output.")
args = parser.parse_args()


if __name__ == "__main__":

    # datasets_torch, cate_attrib_book, dense_feat_index, sparse_feat_index = load_data('./adult-dataset/adult.csv', val_size=0.2, test_size=0.2, run_num=0) #
    # x_train, y_train, z_train, x_val, y_val, z_val, x_test, y_test, z_test = datasets_torch
    # print(x_train.shape)

    # print(cate_attrib_book)

    if args.softmax:
        model_checkpoint_fname = "./adult-dataset/model_softmax_adult_m_1_r_0.pth.tar"
    else:
        model_checkpoint_fname = "/opt/data/private/wzf/u_shapley/backup/SHEAR-main/adult-dataset/model_adult_m_1_r_0.pth.tar"

    checkpoint = torch.load(model_checkpoint_fname)

    # checkpoint = torch.load("./adult-dataset/model_adult_m_1_l_5_r_0.pth.tar")
    # print(checkpoint)
    dense_feat_index  = checkpoint["dense_feat_index"]
    sparse_feat_index = checkpoint["sparse_feat_index"]
    cate_attrib_book  = checkpoint["cate_attrib_book"]

    model = mlp(input_dim=checkpoint["input_dim"],
                output_dim=checkpoint["output_dim"],
                layer_num=checkpoint["layer_num"],
                hidden_dim=checkpoint["hidden_dim"],
                activation=checkpoint["activation"])

    model.load_state_dict(checkpoint["state_dict"])

    model_for_shap = Model_for_shap(model, dense_feat_index, sparse_feat_index, cate_attrib_book)

    x_test = checkpoint["test_data_x"]#为什么形状是[]
    y_test = checkpoint["test_data_y"]
    z_test = checkpoint["test_data_z"]
    shapley_value_gt = checkpoint["test_shapley_value"]
    shapley_ranking_gt = checkpoint["test_shapley_ranking"]
    reference = checkpoint["reference"]

    N = shapley_value_gt.shape[0]
    data_loader = DataLoader(TensorDataset(x_test[0:N], y_test[0:N], z_test[0:N], shapley_value_gt, shapley_ranking_gt),
                             batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

    print(x_test[:, 0:5])
    print(z_test[:, 0:5])

    error_matrix = grad_estimation(model.forward_1, data_loader, reference,
                                                 dense_feat_index, sparse_feat_index, cate_attrib_book)

    save_checkpoint(model_checkpoint_fname,
                    round_index=checkpoint["round_index"],
                    state_dict=checkpoint["state_dict"],
                    layer_num=checkpoint["layer_num"],
                    input_dim=checkpoint["input_dim"],
                    hidden_dim=checkpoint["hidden_dim"],
                    output_dim=checkpoint["output_dim"],
                    activation=checkpoint["activation"],
                    test_data_x=checkpoint["test_data_x"],
                    test_data_y=checkpoint["test_data_y"],
                    test_data_z=checkpoint["test_data_z"],
                    reference=checkpoint["reference"],
                    test_shapley_value=checkpoint["test_shapley_value"],
                    test_shapley_ranking=checkpoint["test_shapley_ranking"],
                    cate_attrib_book=cate_attrib_book,
                    dense_feat_index=dense_feat_index,
                    sparse_feat_index=sparse_feat_index,
                    error_matrix = error_matrix,
                    )

