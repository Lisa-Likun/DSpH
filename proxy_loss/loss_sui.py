from argparse import Namespace

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def proxy_synthesis(input_l2, proxy_l2, target, ps_rate, ps_mu):
    """
    origin code, which should be used with sampler() & redist()
    :param input_l2: [batch_size, dims] l2-normalized embedding features
    :param proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    :param target: [batch_size] note that adjacent labels should be different (e.g., [0,1,2,3,4,5,...])
    :param ps_rate: lambda for linear interpolation
    :param ps_mu: generation ratio (# of synthetics / batch_size)
    """
    input_list = [input_l2]
    proxy_list = [proxy_l2]
    target_list = [target]
    # linear interpolation function
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
    target = torch.cat(target_list, dim=0)[:embed_size]

    input_l2 = F.normalize(input_large, p=2, dim=1)
    proxy_l2 = F.normalize(proxy_large, p=2, dim=1)

    return input_l2, proxy_l2, target


def my_proxy_synthesis1(embeddings, proxies, labels, ps_rate, ps_mu):
    """
    add multi-hot labels support
    same result as proxy_synthesis() for single-labelled dataset
    :param embeddings: [batch_size, dims] l2-normalized embedding features
    :param proxies: [n_classes, dims] l2-normalized proxies
    :param labels: [batch_size, n_classes] multi-hot codec labels
    :param ps_rate: lambda for linear interpolation
    :param ps_mu: generation ratio (# of synthetics / batch_size)
    """
    input_list = [embeddings]
    proxy_list = [proxies]

    diff_idx = torch.triu(labels @ labels.T == 0, diagonal=1).nonzero()

    # linear interpolation function
    input_aug = ps_rate * embeddings[diff_idx[:, 0]] + (1.0 - ps_rate) * embeddings[diff_idx[:, 1]]
    proxy_aug = ps_rate * get_centroids(labels[diff_idx[:, 0]], proxies) + (1.0 - ps_rate) * get_centroids(
        labels[diff_idx[:, 1]], proxies
    )
    input_list.append(input_aug)
    proxy_list.append(proxy_aug)

    # μ x batch_size = #of synthetics
    n_synthetics = int(embeddings.shape[0] * ps_mu)
    # print("n_synthetics", n_synthetics)
    assert n_synthetics <= diff_idx.shape[0]
    # pad(left, right, top, bottom)
    label_list = [F.pad(input=labels, pad=(0, n_synthetics, 0, 0), mode="constant", value=0)]
    pseudo_labels = torch.eye(n_synthetics, device=labels.device)
    pseudo_labels = F.pad(input=pseudo_labels, pad=(labels.shape[1], 0, 0, 0), mode="constant", value=0)
    label_list.append(pseudo_labels)
    embed_size = int(embeddings.shape[0] + n_synthetics)
    proxy_size = int(proxies.shape[0] + n_synthetics)
    input_large = torch.cat(input_list, dim=0)[:embed_size, :]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size, :]
    labels = torch.cat(label_list, dim=0)[:embed_size]

    embeddings = F.normalize(input_large, p=2, dim=1)
    proxies = F.normalize(proxy_large, p=2, dim=1)

    return embeddings, proxies, labels


def get_centroids(labels, proxies):
    # centroids = (labels.unsqueeze(2) * proxies.unsqueeze(0)).sum(1) / labels.sum(1).unsqueeze(1)
    labels = labels.to(0)
    labels = labels.double()
    proxies = proxies.double()
    centroids = (labels @ proxies) / labels.sum(1).unsqueeze(1)
    return centroids


def my_proxy_synthesis2(embeddings, proxies, labels, ps_rate, ps_mu):
    """
    improved from my_proxy_synthesis1()
    :param embeddings: [batch_size, dims] l2-normalized embedding features
    :param proxies: [n_classes, dims] l2-normalized proxies
    :param labels: [batch_size, n_classes] multi-hot codec labels
    :param ps_rate: lambda for linear interpolation
    :param ps_mu: generation ratio (# of synthetics / batch_size)
    """
    input_list = [embeddings]
    # centroids = get_centroids(labels, proxies)
    # centroid_list = [centroids]

    diff_idx = torch.triu(labels @ labels.T == 0, diagonal=1).nonzero()#返回输入张量的右上角三角部分，下三角元素都为0

    # linear interpolation function
    input_aug = ps_rate * embeddings[diff_idx[:, 0]] + (1.0 - ps_rate) * embeddings[diff_idx[:, 1]]#[:, 1]  第二个维度上第二行的所有元素
    # centroid_aug = ps_rate * centroids[diff_idx[:, 0]] + (1.0 - ps_rate) * centroids[diff_idx[:, 1]]
    input_list.append(input_aug)
    # centroid_list.append(centroid_aug)

    # μ x batch_size = #of synthetics
    n_synthetics = int(embeddings.shape[0] * ps_mu)
    # print("n_synthetics", n_synthetics)
    assert n_synthetics <= diff_idx.shape[0]
    # pad(left, right, top, bottom)
    label_list = [F.pad(input=labels, pad=(0, n_synthetics, 0, 0), mode="constant", value=0)]
    pseudo_labels = torch.eye(n_synthetics, device=labels.device)#torch.eye()的用法是生成对角线为1其余全为0的矩阵
    pseudo_labels = F.pad(input=pseudo_labels, pad=(labels.shape[1], 0, 0, 0), mode="constant", value=0)
    label_list.append(pseudo_labels)
    embed_size = int(embeddings.shape[0] + n_synthetics)
    # centroid_size = int(centroids.shape[0] + n_synthetics)
    # input_large = torch.cat(input_list, dim=0)[:embed_size, :]
    centroid_large = torch.cat(centroid_list, dim=0)[:centroid_size, :]
    labels = torch.cat(label_list, dim=0)[:embed_size]

    embeddings = F.normalize(input_large, p=2, dim=1)
    # centroids = F.normalize(centroid_large, p=2, dim=1)

    return embeddings, labels


class Norm_SoftMax(nn.Module):
    def __init__(self, args: Namespace, ps_rate=None):
        super(Norm_SoftMax, self).__init__()
        self.scale = args.scale
        self.ps_mu = args.ps_mu
        self.ps_alpha = args.ps_alpha
        self.method = args.method
        self.ps_rate = ps_rate
        self.proxies = nn.Parameter(torch.Tensor(args.n_classes, args.n_bits))
        nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)

        if self.ps_rate is None:
            # beta distribution
            ps_rate = np.random.beta(self.ps_alpha, self.ps_alpha)  # 从beta分布中任意取一个数
        else:
            ps_rate = self.ps_rate

        if len(labels.shape) == 1:
            embeddings, proxies, targets = proxy_synthesis(embeddings, proxies, labels, ps_rate, self.ps_mu)
            cos_sim = embeddings.matmul(proxies.t())
            logits = self.scale * cos_sim  # BS x C+
            # print("logits.shape", logits.shape)
            # print("targets.shape", targets.shape)
            # print("--> test", (-nn.LogSoftmax(dim=1)(logits[:, targets])).sum(1).mean().item())
            print("--> same", (-nn.LogSoftmax(dim=1)(logits))[range(len(logits)), targets].mean().item())
            return F.cross_entropy(logits, targets)

        # multi-hot codec label
        if self.method == 1:
            embeddings, proxies, labels = my_proxy_synthesis1(embeddings, proxies, labels, ps_rate, self.ps_mu)
            proxies = proxies.float()
            cos_sim = embeddings.matmul(proxies.t())
            logits = self.scale * cos_sim
            labels = labels.to(0)
            return F.cross_entropy(logits, labels).to(0)
        else:
            embeddings, centroids, labels = my_proxy_synthesis2(embeddings, proxies, labels, ps_rate, self.ps_mu)
            print("self.method == 2")

            ##########################################################
            cos_sim = embeddings.matmul(centroids.t())
            logits = self.scale * cos_sim
            return F.cross_entropy(logits, (labels @ labels.T > 0).float())
            ##########################################################


if __name__ == "__main__":
    batch_size = 64
    n_bits = 16
    n_classes = 24

    _embeddings = torch.randn(batch_size, n_bits).cuda()
    _labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float().cuda()
    _labels = F.pad(_labels, (0, n_classes - 3, 0, 0), "constant", 0)
    _targets = torch.argmax(_labels, dim=1).int()

    args = Namespace(n_bits=n_bits, n_classes=n_classes, scale=23.0, ps_alpha=0.40, ps_mu=0.8, method=1)

    criterion = Norm_SoftMax(args, ps_rate=0.5).cuda()

    print("gitcode", criterion(_embeddings, _targets).item())
    print("-" * 10)
    print("method1", criterion(_embeddings, _labels).item())
    print("-" * 10)
    criterion.method = 2
    print("method2", criterion(_embeddings, _labels).item())
