import abc
import torch
import torch.nn as nn

import typing
from . import function
from ..dependency import Group
from .._helpers import _FlattenIndexMapping
from .. import ops
import math
import numpy as np
from collections import OrderedDict
from ..utils.compute_mat_grad import ComputeMatGrad
import logging

__all__ = [
    # Base Class
    "Importance",

    # Basic Group Importance
    "GroupNormImportance",
    "GroupTaylorImportance",
    "GroupHessianImportance",

    # Aliases
    "MagnitudeImportance",
    "TaylorImportance",
    "HessianImportance",

    # Other Importance
    "BNScaleImportance",
    "LAMPImportance",
    "RandomImportance",
    "ShapleyImportance"
]

class Importance(abc.ABC):
    """ Estimate the importance of a tp.Dependency.Group, and return an 1-D per-channel importance score.

        It should accept a group as inputs, and return a 1-D tensor with the same length as the number of channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.

        Example:
            ```python
            DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
            group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
            scorer = MagnitudeImportance()    
            imp_score = scorer(group)    
            #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
            min_score = imp_score.min() 
            ``` 
    """
    @abc.abstractclassmethod
    def __call__(self, group: Group) -> torch.Tensor: 
        raise NotImplementedError


class GroupNormImportance(Importance):
    """ A general implementation of magnitude importance. By default, it calculates the group L2-norm for each channel/dim.
        It supports several variants like:
            - Standard L1-norm of the first layer in a group: MagnitudeImportance(p=1, normalizer=None, group_reduction="first")
            - Group L1-Norm: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean")
            - BN Scaling Factor: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean", target_types=[nn.modules.batchnorm._BatchNorm])

        Args:
            * p (int): the norm degree. Default: 2
            * group_reduction (str): the reduction method for group importance. Default: "mean"
            * normalizer (str): the normalization method for group importance. Default: "mean"
            * target_types (list): the target types for importance calculation. Default: [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]

        Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = GroupNormImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 p: int=2, 
                 group_reduction: str="mean", 
                 normalizer: str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm]):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias

    def _lamp(self, scores): # Layer-adaptive Sparsity for the Magnitude-based Pruning
        """
        Normalizing scheme for LAMP.
        """
        # sort scores in an ascending order
        sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
        # compute cumulative sum
        scores_cumsum_temp = sorted_scores.cumsum(dim=0)
        scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
        scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
        # normalize by cumulative sum
        sorted_scores /= (scores.sum() - scores_cumsum)
        # tidy up and output
        new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
        new_scores[sorted_idx] = sorted_scores
        
        return new_scores.view(scores.shape)
    
    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        elif normalizer.startswith('sentinel'): # normalize the score with the k-th smallest element. e.g. sentinel_0.5 means median normalization
            sentinel = float(normalizer.split('_')[1]) * len(group_importance)
            sentinel = torch.argsort(group_importance, dim=0, descending=False)[int(sentinel)]
            return group_importance / (group_importance[sentinel]+1e-8)
        elif normalizer=='lamp':
            return self._lamp(group_importance)
        else:
            raise NotImplementedError

    def _reduce(self, group_imp: typing.List[torch.Tensor], group_idxs: typing.List[typing.List[int]]):
        if len(group_imp) == 0: return group_imp
        if self.group_reduction == 'prod':
            reduced_imp = torch.ones_like(group_imp[0])
        elif self.group_reduction == 'max':
            reduced_imp = torch.ones_like(group_imp[0]) * -99999
        else:
            reduced_imp = torch.zeros_like(group_imp[0])

        for i, (imp, root_idxs) in enumerate(zip(group_imp, group_idxs)):
            imp = imp.to(reduced_imp.device)
            if self.group_reduction == "sum" or self.group_reduction == "mean":
                reduced_imp.scatter_add_(0, torch.tensor(root_idxs, device=imp.device), imp) # accumulated importance
            elif self.group_reduction == "max": # keep the max importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                selected_imp = torch.maximum(input=selected_imp, other=imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == "prod": # product of importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                torch.mul(selected_imp, imp, out=selected_imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == 'first':
                if i == 0:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction == 'gate':
                if i == len(group_imp)-1:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction is None:
                reduced_imp = torch.stack(group_imp, dim=0) # no reduction
            else:
                raise NotImplementedError
        
        if self.group_reduction == "mean":
            reduced_imp /= len(group_imp)
        return reduced_imp
    
    @torch.no_grad()
    def __call__(self, group: Group):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias.data[idxs].abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                
                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # BatchNorm
            ####################
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            ####################
            # LayerNorm
            ####################
            elif prune_fn == function.prune_layernorm_out_channels:

                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class BNScaleImportance(GroupNormImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519

    Example:
    
        It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
        
        ```python
            DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
            group = DG.get_pruning_group( model.bn1, tp.prune_batchnorm_out_channels, idxs=[2, 6, 9] )    
            scorer = BNScaleImportance()    
            imp_score = scorer(group)    
            #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
            min_score = imp_score.min() 
        ``` 

    """

    def __init__(self, group_reduction='mean', normalizer='mean'):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer, bias=False, target_types=(nn.modules.batchnorm._BatchNorm,))


class LAMPImportance(GroupNormImportance):
    """Layer-adaptive Sparsity for the Magnitude-based Pruning,
    https://arxiv.org/abs/2010.07611

    Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = LAMPImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='lamp', bias=False):
        assert normalizer == 'lamp'
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)


class FPGMImportance(GroupNormImportance):
    """Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration,
    http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='mean', bias=False):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p)
                # calculate the euclidean distance as similarity
                similar_matrix = torch.cdist(local_imp.unsqueeze(0), local_imp.unsqueeze(0), p=2).squeeze(0)
                similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)
                group_imp.append(similar_sum)
                group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)

                local_imp = w.abs().pow(self.p)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                local_imp = local_imp[idxs]
                similar_matrix = torch.cdist(local_imp.unsqueeze(0), local_imp.unsqueeze(0), p=2).squeeze(0)
                similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)
                group_imp.append(similar_sum)
                group_idxs.append(root_idxs)

            # FPGMImportance should not care about BatchNorm and LayerNorm

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class RandomImportance(Importance):
    """ Random importance estimator
    Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = RandomImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs))


class GroupTaylorImportance(GroupNormImportance):
    """ Grouped first-order taylor expansion of the loss function.
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf

        Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                loss = loss_fn(model(inputs), labels)
                loss.backward() # compute gradients
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = GroupTaylorImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 multivariable:bool=False, 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue
            
            # Conv/Linear Output
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    dw = layer.weight.grad.data.transpose(1, 0)[
                        idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    dw = layer.weight.grad.data[idxs].flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    b = layer.bias.data[idxs]
                    db = layer.bias.grad.data[idxs]
                    local_imp = (b * db).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv/Linear Input
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                    dw = (layer.weight.grad).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                    dw = (layer.weight.grad).transpose(0, 1).flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                
                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                local_imp = local_imp[idxs]

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class OBDCImportance(GroupNormImportance):
    """EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis:
       http://proceedings.mlr.press/v97/wang19g/wang19g.pdf
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear],
                 num_classes=100):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self.A, self.DS = {}, {}
        self.Fisher = {}
        self.MatGradHandler = ComputeMatGrad()
        self.steps = 0
        self.eps = 1e-10
        self.modules = []
        self.num_classes = num_classes
        self.known_modules = {'Linear', 'Conv2d'}
    
    def step(self):
        with torch.no_grad():
            for m in self.modules:
                A, DS = self.A[m], self.DS[m]
                grad_mat = self.MatGradHandler(A, DS, m)
                grad_mat *= DS.size(0)
                if self.steps == 0:
                    self.Fisher[m] = grad_mat.new(grad_mat.size()[1:]).fill_(0)
                self.Fisher[m] += (grad_mat.pow_(2)).sum(0)
                self.A[m] = None
                self.DS[m] = None
        self.steps += 1

    def adjust_fisher(self, group, idxs):
        for i, (dep, id) in enumerate(group):
            layer = dep.target.module
            if layer in self.modules:
                if layer.weight.grad is not None:
                    shape = layer.weight.shape
                    if isinstance(layer, nn.modules.conv._ConvNd):
                        kernel_size = shape[2]*shape[3]
                    else:
                        kernel_size = 1
                    indices_to_keep = list(range(self.Fisher[layer].shape[1]))
                    for idx in idxs:
                        indices_to_keep = [i for i in indices_to_keep if not (idx*kernel_size <= i < (idx+1)*kernel_size)]
                    self.Fisher[layer] = torch.index_select(self.Fisher[layer], 1, torch.LongTensor(indices_to_keep).to(self.Fisher[layer].device))
            

    def _rm_hooks(self, model):
        for m in self.modules:
            m._backward_hooks = OrderedDict()
            m._forward_pre_hooks = OrderedDict()

    def _save_input(self, module, input):
        self.A[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        self.DS[module] = grad_output[0].data

    def _prepare_model(self, model, pruner):
        for group in pruner.DG.get_all_groups(ignored_layers=pruner.ignored_layers, root_module_types=pruner.root_module_types): 
            group = pruner._downstream_node_as_root_if_attention(group)
            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                if isinstance(layer, tuple(self.target_types)) and dep.handler in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    self.modules.append(layer)
                    layer.register_forward_pre_hook(self._save_input)
                    layer.register_backward_hook(self._save_grad_output)

    def _clear_buffer(self):
        self.Fisher = {}
        self.modules = []
        self.steps = 0
    
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)) or (isinstance(layer, torch.nn.Linear) and layer.out_features == self.num_classes):
                continue
            F_diag = (self.Fisher[layer] / self.steps + self.eps)
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                    local_imp = (w ** 2 * F_diag).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    local_imp = (b ** 2 * F_diag).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class GroupHessianImportance(GroupNormImportance):
    """Grouped Optimal Brain Damage:
       https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html

       Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                scorer = GroupHessianImportance()   
                scorer.zero_grad() # clean the acuumulated gradients if necessary
                loss = loss_fn(model(inputs), labels, reduction='none') # compute loss for each sample
                for l in loss:
                    model.zero_grad() # clean the model gradients
                    l.backward(retain_graph=True) # compute gradients for each sample
                    scorer.accumulate_grad(model) # accumulate gradients of each sample
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self._accu_grad = {}
        self._counter = {}

    def zero_grad(self):
        self._accu_grad = {}
        self._counter = {}

    def accumulate_grad(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self._accu_grad:
                    self._accu_grad[param] = param.grad.data.clone().pow(2)
                else:
                    self._accu_grad[param] += param.grad.data.clone().pow(2)
                
                if name not in self._counter:
                    self._counter[param] = 1
                else:
                    self._counter[param] += 1
    
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []

        if len(self._accu_grad) > 0: # fill gradients so that we can re-use the implementation for Taylor
            for p, g in self._accu_grad.items():
                p.grad.data = g / self._counter[p]
            self.zero_grad()

        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                        h = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                        h = layer.weight.grad.data[idxs].flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    h = layer.bias.grad.data[idxs]
                    local_imp = (b**2 * h)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = (layer.weight).flatten(1)
                        h = (layer.weight.grad).flatten(1)
                    else:
                        w = (layer.weight).transpose(0, 1).flatten(1)
                        h = (layer.weight.grad).transpose(0, 1).flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                        local_imp = local_imp.repeat(layer.groups)
                    local_imp = local_imp[idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None and layer.bias.grad is None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


# Aliases
class MagnitudeImportance(GroupNormImportance):
    pass

class TaylorImportance(GroupTaylorImportance):
    pass

class HessianImportance(GroupHessianImportance):
    pass



from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.activation import ReLU, ReLU6, RReLU, LeakyReLU, Sigmoid, Softplus, Tanh

import logging

from abc import ABC, abstractmethod

SUPPORTED_OUT_PRUNING_MODULES = [nn.Linear, _ConvNd]
ACTIVATIONS = [ReLU, ReLU6, RReLU, LeakyReLU, Sigmoid, Softplus, Tanh]
from torchpruner.attributions import ShapleyAttributionMetric

class ShapleyImportance(GroupNormImportance,ShapleyAttributionMetric):

    def __init__(self,
                 model: nn.Module,
                 criterion,
                 data_gen ,
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 multivariable:bool=False, 
                 bias=False,
                 sv_samples=1,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias
        

        self.samples = sv_samples
        self.model = model
        self.data_gen = data_gen
        self.device =torch.device("cuda" )
        self.criterion=criterion
        self.reduction = "mean"
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue
            
            # Conv/Linear Output
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                local_imp=self.run(layer)
                local_imp= torch.tensor(local_imp)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)


                # if hasattr(layer, "transposed") and layer.transposed:
                #     w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                #     dw = layer.weight.grad.data.transpose(1, 0)[
                #         idxs].flatten(1)
                # else:
                #     w = layer.weight.data[idxs].flatten(1)
                #     dw = layer.weight.grad.data[idxs].flatten(1)
                # if self.multivariable:
                #     local_imp = (w * dw).sum(1).abs()
                # else:
                #     local_imp = (w * dw).abs().sum(1)
                # group_imp.append(local_imp)
                # group_idxs.append(root_idxs)

                # if self.bias and layer.bias is not None:
                #     b = layer.bias.data[idxs]
                #     db = layer.bias.grad.data[idxs]
                #     local_imp = (b * db).abs()
                #     group_imp.append(local_imp)
                #     group_idxs.append(root_idxs)
                    
        #     # Conv/Linear Input
        #     elif prune_fn in [
        #         function.prune_conv_in_channels,
        #         function.prune_linear_in_channels,
        #     ]:
        #         if hasattr(layer, "transposed") and layer.transposed:
        #             w = (layer.weight).flatten(1)
        #             dw = (layer.weight.grad).flatten(1)
        #         else:
        #             w = (layer.weight).transpose(0, 1).flatten(1)
        #             dw = (layer.weight.grad).transpose(0, 1).flatten(1)
        #         if self.multivariable:
        #             local_imp = (w * dw).sum(1).abs()
        #         else:
        #             local_imp = (w * dw).abs().sum(1)
                
        #         # repeat importance for group convolutions
        #         if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
        #             local_imp = local_imp.repeat(layer.groups)
        #         local_imp = local_imp[idxs]

        #         group_imp.append(local_imp)
        #         group_idxs.append(root_idxs)

        #     # BN
        #     elif prune_fn == function.prune_groupnorm_out_channels:
        #         # regularize BN
        #         if layer.affine:
        #             w = layer.weight.data[idxs]
        #             dw = layer.weight.grad.data[idxs]
        #             local_imp = (w*dw).abs()
        #             group_imp.append(local_imp)
        #             group_idxs.append(root_idxs)

        #             if self.bias and layer.bias is not None:
        #                 b = layer.bias.data[idxs]
        #                 db = layer.bias.grad.data[idxs]
        #                 local_imp = (b * db).abs()
        #                 group_imp.append(local_imp)
        #                 group_idxs.append(root_idxs)
            
        #     # LN
        #     elif prune_fn == function.prune_layernorm_out_channels:
        #         if layer.elementwise_affine:
        #             w = layer.weight.data[idxs]
        #             dw = layer.weight.grad.data[idxs]
        #             local_imp = (w*dw).abs()
        #             group_imp.append(local_imp)
        #             group_idxs.append(root_idxs)
        #             if self.bias and layer.bias is not None:
        #                 b = layer.bias.data[idxs]
        #                 db = layer.bias.grad.data[idxs]
        #                 local_imp = (b * db).abs()
        #                 group_imp.append(local_imp)
        #                 group_idxs.append(root_idxs)
        # if len(group_imp) == 0: # skip groups without parameterized layers
        #     return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)

        return group_imp















    #     self.mask_indices = []


    # def _find_best_module_for_attributions(model, module):
    #     """
    #     Given a Linear or Convolutional module, this method checks if the module
    #     if followed by either BatchNormalization and/or a non-linearity.
    #     In this case, returns the last of these modules as better location
    #     to compute attributions on.
    #     :param model: PyTorch module containing module
    #     :param module: PyTorch Linear or Conv module
    #     :return:
    #     """
    #     modules = list(model.modules())
    #     try:
    #         current_idx = modules.index(module)
    #         eval_module = module
    #         for next_module in modules[current_idx+1:]:
    #             if isinstance(next_module, nn.modules.batchnorm._BatchNorm):
    #                 print(f"BatchNorm detected: shifting evaluation after {next_module}")
    #                 eval_module = next_module
    #             elif any([isinstance(next_module, t) for t in torch.nn.modules.activation.ReLU]):
    #                 print(f"Activation detected: shifting evaluation after {next_module}")
    #                 eval_module = next_module
    #             else:
    #                 return eval_module
    #     except ValueError:
    #         logging.error("Provided module is not in model")
    #     return module


    # def _run(self, module, **kwargs):
    #     assert any(
    #         [isinstance(module, t) for t in self.target_types]
    #     ), f"Attributions can be computed only for the following modules {self.target_types}"
    #     return self._find_evaluation_module(module, **kwargs)

    # def _find_evaluation_module(self, module, find_best_evaluation_module=False):
    #     if find_best_evaluation_module is True:
    #         return self._find_best_module_for_attributions(self.model, module)
    #     else:
    #         return module

    # def run(self, module, sv_samples=None, **kwargs):
    #     module = self._run(module, **kwargs)
    #     sv_samples = sv_samples if sv_samples is not None else self.samples
    #     if hasattr(self.model, "forward_partial"):
    #         result = self._run_module_with_partial(module, sv_samples)
    #     else:
    #         logging.warning("Consider adding a 'forward_partial' method to your model to speed-up Shapley values "
    #                         "computation")
    #         result = self._run_module(module, sv_samples)
    #     return result

    # def _run_module_with_partial(self, module, sv_samples):
    #     """
    #     Implementation of Shapley value monte carlo sampling for models
    #     that provides a `forward_partial` function. This is significantly faster
    #     than run_module(), as it only runs the forward pass on the necessary modules.
    #     """
    #     d = len(self.data_gen.dataset)
    #     sv = None
    #     permutations = None
    #     c = 0

    #     with torch.no_grad():
    #         for idx, (x, y) in enumerate(self.data_gen):
    #             x, y = x.to(self.device), y.to(self.device)
    #             original_z, _ = self.run_forward_partial(x, to_module=module)
    #             _, original_loss = self.run_forward_partial(original_z, y_true=y, from_module=module)
    #             n = original_z.shape[1]  # prunable dimension
    #             if permutations is None:
    #                 # Keep the same permutations for all batches
    #                 permutations = [np.random.permutation(n) for _ in range(sv_samples)]
    #             if sv is None:
    #                 sv = np.zeros((d, n))

    #             for j in range(sv_samples):
    #                 loss = original_loss.detach().clone()
    #                 z = original_z.clone().detach()

    #                 for i in permutations[j]:
    #                     z.index_fill_(1, torch.tensor(np.array([i])).long().to(self.device), 0.0)
    #                     _, new_loss = self.run_forward_partial(z, y_true=y, from_module=module)
    #                     delta = new_loss - loss
    #                     n = delta.shape[0]
    #                     sv[c:c+n, i] += (delta / sv_samples).squeeze().detach().cpu().numpy()
    #                     loss = new_loss
    #             c += n

    #         return self.aggregate_over_samples(sv)

    # def _run_module(self, module, samples):
    #     """
    #     Implementation of Shapley value monte carlo sampling.
    #     No further changes to the model are necessary but this can be quite slow.
    #     See run_module_with_partial() for a faster version that uses partial evaluation.
    #     """
    #     with torch.no_grad():
    #         self.mask_indices = []
    #         handle = module.register_forward_hook(self._forward_hook())
    #         original_loss = self.run_all_forward()
    #         n = module._tp_prune_dim  # output dimension
    #         sv = np.zeros((original_loss.shape[0], n))

    #         for j in range(samples):
    #             # print (f"Sample {j}")
    #             self.mask_indices = []
    #             loss = original_loss.detach().clone()
    #             for i in np.random.permutation(n):
    #                 self.mask_indices.append(i)
    #                 new_loss = self.run_all_forward()
    #                 sv[:, i] += ((new_loss - loss) / samples).squeeze().detach().cpu().numpy()
    #                 loss = new_loss

    #         handle.remove()
    #         return self.aggregate_over_samples(sv)

    # def _forward_hook(self):
    #     def _hook(module, _, output):
    #         module._tp_prune_dim = output.shape[1]
    #         return output.index_fill_(
    #             1, torch.tensor(self.mask_indices).long().to(self.device), 0.0,
    #         )

    #     return _hook
       