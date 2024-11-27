import torch
import torch.nn as nn
import torch_pruning as tp
import random
import numpy as np
from tqdm import tqdm


class Pruner(object):
    def __init__(self, arch_name, model, strategy, pruning_percentage, config,val_loader,criterion):
        self.model = model
        self.strategy = strategy
        self.pruning_percentage = pruning_percentage
        self.arch_name = arch_name

        self.get_channel_idx = None
        if strategy == 'l1':
            self.get_channel_idx = self.__importance_l1
        elif strategy == 'l2':
            self.get_channel_idx = self.__importance_l2
        elif strategy == 'random':
            self.get_channel_idx = self.__importance_random
        elif strategy == 'entropy':
            self.get_channel_idx = self.__importance_entropy
        elif strategy == 'rank':
            self.get_channel_idx = self.__importance_rank
        elif strategy == 'css':
            self.get_channel_idx = self.__importance_css

        self.prune_arch = self.__prune
        self.config = config
        self.val_loader = val_loader
        self.criterion = criterion
    def exec(self):
        self.prune_arch()

    def __importance_l1(self, module):
        total_parameters = len(module.weight)

        l1_norm = torch.norm(module.weight.view(total_parameters, -1), p=1, dim=1)

        n_to_prune = int(self.pruning_percentage * total_parameters)
        n_remain = max(int(total_parameters - n_to_prune), 1)
        n_to_prune = max(total_parameters - n_remain, 0)

        if n_to_prune == 0:
            return []

        threshold = torch.kthvalue(l1_norm, k=n_to_prune).values
        indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()

        return indices

    def __importance_l2(self, module):
        total_parameters = len(module.weight)

        l2_norm = torch.norm(module.weight.view(total_parameters, -1), p=2, dim=1)

        n_to_prune = int(self.pruning_percentage * total_parameters)
        n_remain = max(int(total_parameters - n_to_prune), 1)
        n_to_prune = max(total_parameters - n_remain, 0)

        if n_to_prune == 0:
            return []

        threshold = torch.kthvalue(l2_norm, k=n_to_prune).values
        indices = torch.nonzero(l2_norm <= threshold).view(-1).tolist()

        return indices

    def __importance_entropy(self, module):
        total_parameters = len(module.weight)

        tensor = nn.ReLU(inplace=True)(module.weight.data)
        nd_array = np.array(tensor.cpu())
        logp = np.log(nd_array + 1e-5)
        entropy = -nd_array * logp
        while entropy.ndim >= 2:
            entropy = np.sum(entropy, axis=-1)
        entropy = torch.from_numpy(entropy)

        n_to_prune = int(self.pruning_percentage * total_parameters)
        n_remain = max(int(total_parameters - n_to_prune), 1)
        n_to_prune = max(total_parameters - n_remain, 0)

        if n_to_prune == 0:
            return []

        threshold = torch.kthvalue(entropy, k=n_to_prune).values
        indices = torch.nonzero(entropy <= threshold).view(-1).tolist()

        return indices

    def __importance_rank(self, module):
        total_parameters = len(module.weight)

        tensor = nn.ReLU(inplace=True)(module.weight.data)

        in_channels = tensor.shape[0]
        out_channels = tensor.shape[1]

        rank = torch.tensor(
            [torch.linalg.matrix_rank(tensor[i, j, :, :]).item()
             for i in range(in_channels) for j in range(out_channels)]
        )
        rank = rank.reshape(out_channels, -1).float()
        rank = rank.sum(0)

        n_to_prune = int(self.pruning_percentage * total_parameters)
        n_remain = max(int(total_parameters - n_to_prune), 1)
        n_to_prune = max(total_parameters - n_remain, 0)

        if n_to_prune == 0:
            return []

        threshold = torch.kthvalue(rank, k=n_to_prune).values
        indices = torch.nonzero(rank <= threshold).view(-1).tolist()

        return indices

    def __importance_css(self, module):
        total_parameters = len(module.weight)

        _, _, css = torch.svd(module.weight, some=True)
        while css.ndim >= 2:
            css = torch.sum(css, dim=-1)

        n_to_prune = int(self.pruning_percentage * total_parameters)
        n_remain = max(int(total_parameters - n_to_prune), 1)
        n_to_prune = max(total_parameters - n_remain, 0)

        if n_to_prune == 0:
            return []

        threshold = torch.kthvalue(css, k=n_to_prune).values
        indices = torch.nonzero(css <= threshold).view(-1).tolist()

        return indices

    def __importance_random(self, module):
        total_parameters = len(module.weight)

        n_to_prune = int(self.pruning_percentage * total_parameters)
        n_remain = max(int(total_parameters - n_to_prune), 1)
        n_to_prune = max(total_parameters - n_remain, 0)

        if n_to_prune == 0:
            return []

        indices = random.sample(list(range(total_parameters)), k=n_to_prune)

        return indices

    def __prune_unext(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=torch.randn([1, 3, 224, 224]).cuda())
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                try:
                    indices = self.get_channel_idx(module)
                    group = DG.get_pruning_group(module, tp.prune_conv_out_channels,
                                                 idxs=indices)
                    if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
                        group.prune()
                except KeyError:
                    pass

    def __prune_unext_s(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=torch.randn([1, 3, 224, 224]).cuda())
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                try:
                    indices = self.get_channel_idx(module)
                    group = DG.get_pruning_group(module, tp.prune_conv_out_channels,
                                                 idxs=indices)
                    if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
                        group.prune()
                except KeyError:
                    pass

    def __prune_unet(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=torch.randn([1, 3, 224, 224]).cuda())
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                try:
                    indices = self.get_channel_idx(module)
                    group = DG.get_pruning_group(module, tp.prune_conv_out_channels,
                                                 idxs=indices)
                    if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
                        group.prune()
                except KeyError:
                    pass

    def __prune_unetpp(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=torch.randn([1, 3, 224, 224]).cuda())
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                try:
                    indices = self.get_channel_idx(module)
                    group = DG.get_pruning_group(module, tp.prune_conv_out_channels,
                                                 idxs=indices)
                    if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
                        group.prune()
                except KeyError:
                    pass

    def __prune(self):
        example_inputs = torch.randn([1, 3, self.config['input_w'], self.config['input_h']]).cuda()

        imp = None
        ch_sparsity = 0.0

        if 'dsb' in self.config['dataset']:
            if self.arch_name == 'UNet':
                if self.strategy == 'random':
                    imp = tp.importance.RandomImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'l1':
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ch_sparsity = 0.47
                elif self.strategy == 'rank':
                    imp = tp.importance.LAMPImportance()
                    ch_sparsity = 0.65
                elif self.strategy == 'shapley':
                    imp = tp.importance.MagnitudeImportance(p=2)
                    ch_sparsity = 0.5
            elif self.arch_name == 'NestedUNet':
                if self.strategy == 'random':
                    imp = tp.importance.RandomImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'l1':
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ch_sparsity = 0.46
                elif self.strategy == 'rank':
                    imp = tp.importance.LAMPImportance()
                    ch_sparsity = 0.7
                elif self.strategy == 'shapley':
                    imp = tp.importance.MagnitudeImportance(p=2)
                    ch_sparsity = 0.52
        elif 'BUSI' in self.config['dataset']:
            if self.arch_name == 'UNet':
                if self.strategy == 'random':
                    imp = tp.importance.RandomImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'l1':
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ch_sparsity = 0.32
                elif self.strategy == 'rank':
                    imp = tp.importance.LAMPImportance()
                    ch_sparsity = 0.62
                elif self.strategy == 'shapley':
                    imp = tp.importance.ShapleyImportance(model=self.model,data_gen=self.val_loader,criterion=self.criterion)
                    ch_sparsity = 0.33
            elif self.arch_name == 'NestedUNet':
                if self.strategy == 'random':
                    imp = tp.importance.RandomImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'l1':
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ch_sparsity = 0.3
                elif self.strategy == 'rank':
                    imp = tp.importance.LAMPImportance()
                    ch_sparsity = 0.6
                elif self.strategy == 'shapley':
                    imp = tp.importance.ShapleyImportance(model=self.model,data_gen=self.val_loader,criterion=self.criterion)
                    ch_sparsity = 0.3
        elif 'busi_3' in self.config['dataset']:
            if self.arch_name == 'UNet':
                if self.strategy == 'random':
                    imp = tp.importance.RandomImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'l1':
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ch_sparsity = 0.32
                elif self.strategy == 'rank':
                    imp = tp.importance.LAMPImportance()
                    ch_sparsity = 0.6
                elif self.strategy == 'shapley':
                    imp = tp.importance.ShapleyImportance(model=self.model,data_gen=self.val_loader,criterion=self.criterion)
                    ch_sparsity = 0.33
            elif self.arch_name == 'NestedUNet':
                if self.strategy == 'random':
                    imp = tp.importance.RandomImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'l1':
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ch_sparsity = 0.3
                elif self.strategy == 'rank':
                    imp = tp.importance.LAMPImportance()
                    ch_sparsity = 0.6
                elif self.strategy == 'shapley':
                    imp = tp.importance.ShapleyImportance(model=self.model,data_gen=self.val_loader,criterion=self.criterion)
                    ch_sparsity = 0.3
        elif 'isic' in self.config['dataset']:
            if self.arch_name == 'UNet':
                if self.strategy == 'random':
                    imp = tp.importance.RandomImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'l1':
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ch_sparsity = 0.37
                elif self.strategy == 'rank':
                    imp = tp.importance.LAMPImportance()
                    ch_sparsity = 0.6
                elif self.strategy == 'shapley':
                    imp = tp.importance.ShapleyImportance(model=self.model,data_gen=self.val_loader,criterion=self.criterion)
                    ch_sparsity = 0.3
            elif self.arch_name == 'NestedUNet':
                if self.strategy == 'random':
                    imp = tp.importance.RandomImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'l1':
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ch_sparsity = 0.3
                elif self.strategy == 'rank':
                    imp = tp.importance.LAMPImportance()
                    ch_sparsity = 0.3
                elif self.strategy == 'shapley':
                    imp = tp.importance.ShapleyImportance(model=self.model,data_gen=self.val_loader,criterion=self.criterion)
                    ch_sparsity = 0.3

        pruner = tp.pruner.MetaPruner(
            model=self.model,
            example_inputs=example_inputs,
            importance=imp,
            ch_sparsity=ch_sparsity,
            global_pruning=True,
            iterative_steps=1,
            isomorphic=True
        )

        pruner.step()
