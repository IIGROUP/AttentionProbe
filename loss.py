import os
import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def kdloss(y, teacher_scores, T=4):
    # weights = weights.unsqueeze(1)
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, reduce=False)
    loss = torch.sum(l_kl) / y.shape[0]
    return loss * (T**2)

def robust_kdloss(y, teacher_scores, weights, T=4):
    weights = weights.unsqueeze(1)
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, reduce=False)
    loss = torch.sum(l_kl*weights) / y.shape[0]
    return loss * (T**2)

def csloss(y, uniform):
    # y: (B, N-1), uniform: (B, N-1)
    loss = torch.bmm(y.unsqueeze(1),uniform.unsqueeze(2)).squeeze()/(torch.sqrt(torch.sum(torch.pow(y, 2), dim=1))*torch.sqrt(torch.sum(torch.pow(uniform, 2), dim=1)))
    return loss 


def patch_attention_probe_loss(feature_T, feature_S):
    # feature_T: (num_layers, bs, N, c)
    num_layers = feature_T.shape[0]
    B = feature_T.shape[1]
    N = feature_T.shape[2]

    # feature_T_norm: (num_layers, bs, N, c)
    # feature_S_norm: (num_layers, bs, N, c)
    feature_T_norm = F.normalize(feature_T, p=2, dim=2)
    feature_S_norm = F.normalize(feature_S, p=2, dim=2)

    patch_attn_T = feature_T_norm @ feature_T_norm.transpose(-2, -1)
    patch_attn_S = feature_S_norm @ feature_S_norm.transpose(-2, -1)

    # patch_attn_diff: (num_layers, bs, N-1)
    patch_attn_diff = patch_attn_T[:,:,0,1:] - patch_attn_S[:,:,0,1:]
    patch_attn_loss = (patch_attn_diff * patch_attn_diff).view(-1, 1).sum(0) / (num_layers * B * (N-1))     
    return patch_attn_loss.squeeze()


