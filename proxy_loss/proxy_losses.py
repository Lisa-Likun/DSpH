'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

def proxy_synthesis(input_l2, proxy_l2, target, ps_alpha, ps_mu):
    '''
    input_l2: [batch_size, dims] l2-normalized embedding features
    proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    target: [batch_size] Note that adjacent labels should be different (e.g., [0,1,2,3,4,5,...])
    ps_alpha: alpha for beta distribution
    ps_mu: generation ratio (# of synthetics / batch_size)
    '''
    n_classes = 24
    # print("!!!!!!!!!!!!!!!!!!!!!!input_l2!!!!!!!!!!!!!!!!!!!!!!!", input_l2.size)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!proxy_l2!!!!!!!!!!!!!!!!!!!!", proxy_l2.size)
    # exit()

    input_list = [input_l2]
    proxy_list = [proxy_l2]
    target_deci = []
    ####################################
    target_bi = target.int()
    for i in range(len(target_bi)):
        mid = 0
        for j in range(24):
            if target_bi[i][j] == 1:
                mid += j
        target_deci.append(mid)

    print(target_deci)
    # exit()
    ####################################
    target = target_deci
    target_list = [target]

    ps_rate = np.random.beta(ps_alpha, ps_alpha)

    input_aug = ps_rate * input_l2 + (1.0 - ps_rate) * torch.roll(input_l2, 1, dims=0)
    proxy_aug = ps_rate * proxy_l2[target, :] + (1.0 - ps_rate) * torch.roll(proxy_l2[target, :], 1, dims=0)
    input_list.append(input_aug)
    proxy_list.append(proxy_aug)

    n_classes = proxy_l2.shape[0]
    pseudo_target = torch.arange(n_classes, n_classes + input_l2.shape[0]).cuda()
    target_list.append(pseudo_target)


    embed_size = int(input_l2.shape[0] * (1.0 + ps_mu))
    proxy_size = int(n_classes + input_l2.shape[0] * ps_mu)
    input_large = torch.cat(input_list, dim=0)[:embed_size, :]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size, :]
    target_list = torch.tensor(target_list)
    # print(type(target_list))
    # exit()
    target = torch.cat(target_list, dim=0)[:embed_size]

    input_l2 = F.normalize(input_large, p=2, dim=1)
    proxy_l2 = F.normalize(proxy_large, p=2, dim=1)

    return input_l2, proxy_l2, target


class Norm_SoftMax(nn.Module):
    def __init__(self, input_dim, n_classes, scale=23.0, ps_mu=0.0, ps_alpha=0.0):
        super(Norm_SoftMax, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(24, 16))

        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)



        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)

        sim_mat = input_l2.matmul(proxy_l2.t())

        logits = self.scale * sim_mat

        loss = F.cross_entropy(logits, target)

        return loss


class Proxy_NCA(nn.Module):
    def __init__(self, input_dim, n_classes, scale=10.0, ps_mu=0.0, ps_alpha=0.0):
        super(Proxy_NCA, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(24, 16))

        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)

        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)

        dist_mat = torch.cdist(input_l2, proxy_l2) ** 2
        dist_mat *= self.scale
        pos_target = F.one_hot(target, dist_mat.shape[1]).float()
        loss = torch.mean(torch.sum(-pos_target * F.log_softmax(-dist_mat, -1), -1))

        return loss