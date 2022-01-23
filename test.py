#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models import create_model
import vision_transformer
from functools import partial
# from torchvision.datasets import CIFAR100,ImageFolder,CIFAR10
# import torchvision.transforms as transforms
# import numpy as np
# import math
# # from torch.autograd import Variable
# from resnet import ResNet18,ResNet34
# from torchvision.datasets import CIFAR100,ImageFolder,CIFAR10
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from timm.models import create_model
# import vision_transformer
# from functools import partial

# import pdb
# import numpy as np
# import warnings
# import math

# from PIL import Image
# import requests
# import matplotlib.pyplot as plt
# import torchvision.transforms as T


# acc = 0
# acc_best = 0

# teacher = None





# N = 10
# B = 8
# uniform = torch.ones(B, N-1)/(N-1)
# print(uniform)
# print(uniform.shape)

# a = torch.Tensor([[1,2],[3,4]])
# b = torch.Tensor([[1,2],[3,4]])
# c = torch.bmm(a.unsqueeze(1),b.unsqueeze(2)).squeeze()/(torch.sqrt(torch.sum(torch.pow(a, 2), dim=1))*torch.sqrt(torch.sum(torch.pow(b, 2), dim=1)))
# a1 = torch.pow(a, 2)
# a2 = torch.sum(a1, 1)
# a3 = torch.sqrt(a2)
# b1 = torch.pow(b, 2)
# b2 = torch.sum(b1, 1)
# b3 = torch.sqrt(b2)
# f = torch.bmm(a.unsqueeze(1),b.unsqueeze(2)).squeeze()
# # print(c.squeeze())
# print(a3*b3)
# print(c == f/(a3*b3))

# input = torch.randn(3,3)
# target = torch.LongTensor([0,2,1])
# loss1 = nn.NLLLoss()
# loss2 = nn.CrossEntropyLoss()
# a = loss1(torch.log(F.softmax(input)), target)
# b = loss2(input, target)
# print(a)
# print(b)



# calculate flops and params
# teacher = create_model(
#     'deit_tiny_patch4_28_teacher',
#     pretrained=False,
#     num_classes=1000,
# )
# net = vision_transformer.TeacherVisionTransformer(img_size=28, patch_size=4, in_chans=1, num_classes=10, embed_dim=128, depth=12, num_heads=2, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

# # from torchvision.models import resnet50
# from thop import profile

# input = torch.randn(1, 1, 28, 28)
# flops, params = profile(net, inputs=(input, ))
# print(flops, params)



# get difference set
# a=[x for x in range(1, 50)]
# b=[x for x in range(0, 100)]
# c=list(set(b).difference(set(a)))#b中有而a中没有的 非常高效！
# print(c)

# Accuracy 5

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

# output = torch.Tensor([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.11, 0.23, 0.39, 0.06], [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.11, 0.23, 0.39, 0.06]])
# target = torch.Tensor([8,7])
# acc1, acc5 = accuracy(output, target, topk=(1, 5))
# # print(acc5)
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# patch_size = 16
# patch_size = to_2tuple(patch_size)
# print(patch_size)
# print(*patch_size)


# import torch import torch.nn as nn
# import torch.utils.data as Data
# import torchvision  # 数据库模块
# import matplotlib.pyplot as plt
# # torch.manual_seed(1)  
# # # reproducibleEPOCH = 1  # 训练整批数据次数，训练次数越多，精度越高，为了演示，我们训练5次BATCH_SIZE = 50  
# # # 每次训练的数据集个数LR = 0.001  
# # # 学习效率DOWNLOAD_MNIST = Ture  
# # # 如果你已经下载好了EMNIST数据就设置 False# EMNIST 手写字母 训练集
# # train_data = torchvision.datasets.EMNIST(    root='./data',    train=True,    transform=torchvision.transforms.ToTensor(),    download = DOWNLOAD_MNIST,    split = 'letters' )
# # # EMNIST 手写字母 测试集
# # test_data = torchvision.datasets.EMNIST(    root='./data',    train=False,    transform=torchvision.transforms.ToTensor(),    download=False,    split = 'letters'     )# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)# 每一步 loader 释放50个数据用来学习# 为了演示, 我们测试时提取2000个数据先# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.  test_y = test_data.targets[:2000]#test_x = test_x.cuda() # 若有cuda环境，取消注释#test_y = test_y.cuda() # 若有cuda环境，取消注释



# torchvision.datasets.EMNIST(root: str, split: str, **kwargs: Any)
# root ( string ) –数据集所在EMNIST/processed/training.pt 和 EMNIST/processed/test.pt存在的根目录。split（字符串）   -该数据集具有6个不同的拆分：byclass，bymerge， balanced，letters，digits和mnist。此参数指定使用哪一个。train ( bool , optional )– 如果为 True，则从 中创建数据集training.pt，否则从test.pt.download ( bool , optional ) – 如果为 true，则从 Internet 下载数据集并将其放在根目录中。如果数据集已经下载，则不会再次下载。transform ( callable , optional ) – 一个函数/转换，它接收一个 PIL 图像并返回一个转换后的版本。例如，transforms.RandomCroptarget_transform ( callable , optional ) – 一个接收目标并对其进行转换的函数/转换。
