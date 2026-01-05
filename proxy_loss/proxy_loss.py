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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'



def proxy_synthesis(input_l2, proxy_l2, target, ps_alpha, ps_mu):
    '''
    input_l2: [batch_size, dims] l2-normalized embedding features
    proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    target: [batch_size] Note that adjacent labels should be different (e.g., [0,1,2,3,4,5,...])
    ps_alpha: alpha for beta distribution
    ps_mu: generation ratio (# of synthetics / batch_size)
    '''

    input_list = [input_l2]
    proxy_list = [proxy_l2]
    ####################################
    # target_list = [target]

    target_deci = []
    target_bi = target.int()


    for i in range(len(target_bi)):
        ###############################
        # for j in range(24):
        #     if target_bi[i][j] == 1:
        #         target_deci.append(j)
        #         # print("i,j=", i,j)
        #         break
        #############################
        mid = 0
        for j in range(24):
            if target_bi[i][j] == 1:
                mid += j
        target_deci.append(mid)
        #################################
    target = target_deci
    # print(target)

    target_list = [target]
    target_tensor_qehalf = torch.tensor(target_list).to(0)
    ps_rate = np.random.beta(ps_alpha, ps_alpha)
    input_aug = ps_rate * input_l2 + (1.0 - ps_rate) * torch.roll(input_l2, 1, dims=0)
    proxy_aug = ps_rate * proxy_l2[target_tensor_qehalf, :] + (1.0 - ps_rate) * torch.roll(proxy_l2[target_tensor_qehalf, :], 1, dims=0)
    proxy_aug = proxy_aug.squeeze(0)


    input_list.append(input_aug)

    proxy_list.append(proxy_aug)
    n_classes = proxy_l2.shape[0]
    pseudo_target = torch.arange(n_classes, n_classes + input_l2.shape[0])
    pseudo_target.to(0)
    # print(pseudo_target.device)

    # print("type",type(pseudo_target))
    target_list_hehalf = []
    target_list_hehalf.append(pseudo_target)
    target_tensor_hehalf = torch.stack(target_list_hehalf).to(0)



    embed_size = int(input_l2.shape[0] * (1.0 + ps_mu))

    proxy_size = int(n_classes + input_l2.shape[0] * ps_mu)
    input_large = torch.cat(input_list, dim=0)[:embed_size, :]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size, :]

    ####################################################
    # print(target_tensor_qehalf.device)
    # print(target_tensor_hehalf.device)

    # target_tensor_hehalf = target_tensor_hehalf.to(0)
    target_list_totensor = torch.cat((target_tensor_qehalf,target_tensor_hehalf), dim=0)

    target = target_list_totensor[:embed_size]


    # target = torch.cat(target_list_totensor, dim=0)[:embed_size]
    ####################################################
    # target = torch.cat(target_list, dim=0)[:embed_size]

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
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))

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
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))

        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)


        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)

        # print("the shape of input_l2:::", input_l2.shape)
        # print("the shape of proxy_l2:::", proxy_l2.shape)
        dist_mat = torch.cdist(input_l2, proxy_l2) ** 2
        dist_mat *= self.scale
        pos_target = F.one_hot(target, dist_mat.shape[1]).float()

        # exit()
        pos_target = pos_target.reshape(128,564)
        # dist_mat = dist_mat[:64]
        # print("the shape of pos_target:::", pos_target.shape)
        # print("the shape of dist_mat:::", dist_mat.shape)

        loss = torch.mean(torch.sum(-pos_target * F.log_softmax(-dist_mat, -1), -1))

        return loss
