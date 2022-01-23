#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
cpu_num = 4
import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from torch.autograd import Variable
from resnet import ResNet18,ResNet34
from torchvision.datasets import CIFAR100,ImageFolder,CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timm.models import create_model
import vision_transformer
from loss import kdloss, csloss, patch_attention_probe_loss, robust_kdloss
from utils import accuracy, AverageMeter
from functools import partial
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pdb
import numpy as np
import warnings
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

warnings.filterwarnings('ignore')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_select', type=int, default=600000)
parser.add_argument('--data_cifar', type=str, default='/home/wjh19/database/cifar10/')
parser.add_argument('--data_imagenet', type=str, default='/home/wjh19/database/imagenet/train/')
parser.add_argument('--teacher', default='deit_base_patch4_32_teacher', type=str, metavar='MODEL',
                    help='Name of teacher model to train (default: "regnety_160"')
parser.add_argument('--teacher_dir', type=str, default='/home/wjh19/mage/DFND_DeiT/output/cifar10/teacher/checkpoint.pth')
parser.add_argument('--nb_classes', type=int, default=10, help='number of classes')
parser.add_argument('--lr_S', type=float, default=7.5e-4, help='learning rate')
parser.add_argument('--robust', action='store_true', default=False,
                    help='Robust distillation enabled (if avail)')
parser.add_argument('--attnprobe_sel', action='store_true', default=False,
                    help='Distillation by attention prime enabled (if avail)')
parser.add_argument('--random', action='store_true', default=False,
                    help='Randomly select wild data (if avail)')
parser.add_argument('--attnprobe_dist', action='store_true', default=False,
                    help='Distillation by attention prime enabled (if avail)')
parser.add_argument('--attnlier', type=float, default=0.05, help='weight of attention layer to sample the wild data')
parser.add_argument('--outlier', type=float, default=0.9, help='weight of output layer to sample the wild data')
parser.add_argument('--patchattn', type=float, default=0.8, help='weight of patch attention loss')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100', 'imagenet', 'mnist'])
parser.add_argument('--epochs', type=float, default=800)
parser.add_argument('--output_dir', type=str, default='/home/wjh19/mage/DFND_DeiT/output/cifar10/')
parser.add_argument('--selected_file', type=str, default='/home/wjh19/mage/DFND_DeiT/selected/cifar10/')
parser.add_argument('--schedule', default=[200, 300], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
args,_ = parser.parse_known_args()


acc = 0
acc_best = 0
acc5_best = 0

teacher = None

assert args.teacher_dir, 'need to specify teacher-path when using distillation'
print(f"Creating teacher model: {args.teacher}")
teacher = create_model(
    args.teacher,
    pretrained=False,
    num_classes=args.nb_classes,
)
if args.dataset == 'imagenet':
    embed_dim = 768
    num_heads = 12
    img_size = 224
else:
    embed_dim = 384
    num_heads = 3  
    img_size = 32 

checkpoint = torch.load(args.teacher_dir, map_location='cpu')
teacher.load_state_dict(checkpoint['model'])
teacher.cuda()
teacher.eval()
teacher = nn.DataParallel(teacher)

# teacher = torch.load(args.teacher_dir + 'teacher').cuda()
# teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad = False


def get_class_weight(model, dataloader, num_classes=10, T=1):
    classes_outputs = np.zeros(num_classes)
    
    model.eval()
    if os.path.exists(args.selected_file + 'class_weights.pth'):
        class_weights = torch.load(args.selected_file + 'class_weights.pth')
    else:
        for i,(inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            with torch.set_grad_enabled(False):
                outputs, output_feature_t = model(inputs)
                outputs = F.softmax(outputs/T, dim=1)
                for j in range(inputs.shape[0]):
                    classes_outputs += outputs[j].cpu().data.numpy()
        
        class_weights = 1/classes_outputs
        weights_sum = np.sum(class_weights)
        class_weights /= weights_sum
        class_weights *= num_classes
        torch.save(class_weights, args.selected_file + 'class_weights.pth')
    return class_weights 

def perturb(weight, epsilon=0.1, perturb_num=1):
    weights = []
    weights.append(weight)
    for i in range(perturb_num):
        p = np.random.rand(weight.shape[0]) * epsilon
        weight_new = weight + p
        weights.append(weight_new)
    return weights


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


data_train = ImageFolder(args.data_imagenet, transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    normalize,
]))

data_train_transform = ImageFolder(args.data_imagenet, transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.RandomCrop(img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]))

transform_train = transforms.Compose([
    transforms.RandomCrop(img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_imagenet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


if args.dataset == 'cifar100':
    data_test = CIFAR100(args.data_cifar,
                      train=False,
                      transform=transform_test)
    teacher_acc = torch.tensor([0.7630])
    n_classes = 100
if args.dataset == 'cifar10':
    data_test = CIFAR10(args.data_cifar,
                      train=False,
                      transform=transform_test)
    teacher_acc = torch.tensor([0.9665])
    n_classes = 10
if args.dataset == 'imagenet':
    data_test = ImageFolder(os.path.join(args.data_cifar, 'val'),
                          transform_test_imagenet)
    teacher_acc = torch.tensor([0.8118])
    n_classes = 1000
    

data_test_loader = DataLoader(data_test, batch_size=1000, num_workers=0)

noise_adaptation = torch.nn.Parameter(torch.zeros(n_classes,n_classes-1))

def noisy(noise_adaptation):
    # noise_adaptation_softmax: (n_classes,n_classes-1)
    noise_adaptation_softmax = torch.nn.functional.softmax(noise_adaptation,dim=1) * (1 - teacher_acc)

    # noise_adaptation_layer: (n_classes,n_classes)
    noise_adaptation_layer = torch.zeros(n_classes,n_classes)
    for i in range(n_classes):
        if i == 0:
            noise_adaptation_layer[i] = torch.cat([teacher_acc,noise_adaptation_softmax[i][i:]])
        if i == n_classes-1:
            noise_adaptation_layer[i] = torch.cat([noise_adaptation_softmax[i][:i],teacher_acc])
        else:
            noise_adaptation_layer[i] = torch.cat([noise_adaptation_softmax[i][:i],teacher_acc,noise_adaptation_softmax[i][i:]])

    # noise_adaptation_layer: (n_classes,n_classes)
    return noise_adaptation_layer.cuda()


# net = ResNet18(n_classes).cuda()
if args.dataset == 'imagenet':
    print("Creating student model: deiT_tiny_patch16_224")
    net = vision_transformer.TeacherVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=args.nb_classes, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).cuda()
else:
    print("Creating student model: deiT_xtiny_patch4_32")
    net = vision_transformer.TeacherVisionTransformer(img_size=32, patch_size=4, in_chans=3, num_classes=args.nb_classes, embed_dim=128, depth=12, num_heads=2, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).cuda()
net = torch.nn.DataParallel(net)
criterion = torch.nn.CrossEntropyLoss().cuda()
celoss = torch.nn.CrossEntropyLoss(reduction = 'none').cuda()
# optimizer = torch.optim.SGD(list(net.parameters()), lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr_S, weight_decay=0.025)

optimizer_noise = torch.optim.Adam([noise_adaptation], lr=0.001)
 
data_train_loader_noshuffle = DataLoader(data_train, batch_size=256, shuffle=False, num_workers=8)
    
def identify_attnlier(checkpoint, embed_dim, num_heads):
    value_blk3 = []
    value_blk7 = []
    # pred_list = []
    attn_inputs_blk3 = []
    attn_inputs_blk7 = []
    index = 0
    embed_dim = int(embed_dim/num_heads)
    scale = embed_dim ** -0.5
    teacher.eval()

    # Obtain weights and bias
    # linear_weight_blk_3: (1152, 384). linear_bias_blk_3: (1152)
    linear_weight_blk_3 = checkpoint['model']['blocks.3.attn.qkv.weight'].cuda()
    linear_bias_blk_3 = checkpoint['model']['blocks.3.attn.qkv.bias'].cuda()
    # linear_weight_q_blk_3, linear_weight_k_blk_3, linear_weight_v_blk_3 = torch.split(linear_weight_blk_3, [384, 384, 384], dim=0)
    linear_weight_blk_7 = checkpoint["model"]['blocks.7.attn.qkv.weight'].cuda()
    linear_bias_blk_7 = checkpoint["model"]['blocks.7.attn.qkv.bias'].cuda()
    # linear_weight_q_blk_7, linear_weight_k_blk_7, linear_weight_v_blk_7 = torch.split(linear_weight_blk_7, [384, 384, 384], dim=0)
    hooksadd = [
        teacher.module.blocks[3].attn.register_forward_hook(
            lambda self, input, output: attn_inputs_blk3.append(input)
        ),
        teacher.module.blocks[7].attn.register_forward_hook(
            lambda self, input, output: attn_inputs_blk7.append(input)
        ),
    ]

    for i,(inputs, labels) in enumerate(data_train_loader_noshuffle):
        inputs = inputs.cuda()
        outputs, output_feature = teacher(inputs)

        # calculate input × weights and view the shape
        # B, N, C = 256, 65, 384
        B, N, C = attn_inputs_blk3[index][0].shape
        uniform = (torch.ones(B, N-1)/(N-1)).float().cuda()
        qkv_blk_3 = torch.bmm(attn_inputs_blk3[0][0], linear_weight_blk_3.unsqueeze(0).repeat(B, 1, 1).permute(0, 2, 1)) + linear_bias_blk_3
        qkv_blk_3 = qkv_blk_3.reshape(B, N, 3, num_heads, embed_dim).permute(2, 0, 3, 1, 4)
        q_blk_3, k_blk_3, v_blk_3 = qkv_blk_3[0], qkv_blk_3[1], qkv_blk_3[2]   # make torchscript happy (cannot use tensor as tuple)
        # attn_blk_3: (B, num_heads, N, N) = (256, num_heads, 65, 65)
        attn_blk_3 = (q_blk_3 @ k_blk_3.transpose(-2, -1)) * scale
        attn_blk_3 = attn_blk_3.softmax(dim=-1)
        # attnprime_blk_3: (B, N-1) = (256, 64)
        attnprime_blk_3 = attn_blk_3[:,0,0,1:]

        qkv_blk_7 = torch.bmm(attn_inputs_blk7[0][0], linear_weight_blk_7.unsqueeze(0).repeat(B, 1, 1).permute(0, 2, 1)) + linear_bias_blk_7
        qkv_blk_7 = qkv_blk_7.reshape(B, N, 3, num_heads, embed_dim).permute(2, 0, 3, 1, 4)
        q_blk_7, k_blk_7, v_blk_7 = qkv_blk_7[0], qkv_blk_7[1], qkv_blk_7[2]   # make torchscript happy (cannot use tensor as tuple)
        # attn_blk_7: (B, num_heads, N, N)
        attn_blk_7 = (q_blk_7 @ k_blk_7.transpose(-2, -1)) * scale
        attn_blk_7 = attn_blk_7.softmax(dim=-1)
        # attnprime_blk_7: (B, N-1)
        attnprime_blk_7 = attn_blk_7[:,0,0,1:]

        loss_blk3 = csloss(attnprime_blk_3, uniform)
        loss_blk7 = csloss(attnprime_blk_7, uniform)
        value_blk3.append(loss_blk3.detach().clone())
        value_blk7.append(loss_blk7.detach().clone())
        
        attn_inputs_blk3.clear()
        attn_inputs_blk7.clear()
        print('Considering attnlier of batch %d from the wild massive unlabeled dataset.' % i)
    for hook in hooksadd:
        hook.remove()
    return torch.cat(value_blk3,dim=0), torch.cat(value_blk7,dim=0) 

def identify_outlier():
    value = []
    pred_list = []
    index = 0
    
    teacher.eval()
    for i,(inputs, labels) in enumerate(data_train_loader_noshuffle):
        inputs = inputs.cuda()
        # outputs: (bs, n_classes)
        outputs, output_feature = teacher(inputs)
        # pred: (bs, 1)
        pred = outputs.data.max(1)[1]
        loss = celoss(outputs, pred)
        value.append(loss.detach().clone())
        index += inputs.shape[0]
        pred_list.append(pred)
        
        print('Considering outlier of batch %d from the wild massive unlabeled dataset.' % i)
    return torch.cat(value,dim=0), torch.cat(pred_list,dim=0) 
    
def train(epoch, trainloader, nll, class_weights):
    net.train()
    loss_list, batch_list = [], []
    interval = len(trainloader) // 6
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        optimizer_noise.zero_grad()

        output, output_feature_s = net(images)
        
        output_t, output_feature_t = teacher(images)
        output_t = output_t.detach()
        output_feature_t = output_feature_t.detach()
        pred = output_t.data.max(1)[1]
        preds_t = pred.cpu().data.numpy()
        
        if args.robust:
            for class_weight in class_weights:
                weights = torch.from_numpy(class_weight[preds_t]).float().cuda()
            loss = robust_kdloss(output, output_t, weights)       
        else:
            loss = kdloss(output, output_t)  
               
        output_s = F.softmax(output, dim=1)
        output_s_adaptation = torch.matmul(output_s, noisy(noise_adaptation))
        loss += nll(torch.log(output_s_adaptation), pred)

        if args.attnprobe_dist:
            loss_patch_attn = args.patchattn * patch_attention_probe_loss(output_feature_t, output_feature_s)
            loss += loss_patch_attn

        loss_list.append(loss.data.item())
        batch_list.append(i+1)

        if (i % interval) == 0:
            if args.attnprobe_dist:
                print('Train - Epoch %d, Batch: %d, Loss: %f, Loss_attn: %f' % (epoch, i, loss.data.item(), loss_patch_attn.data.item()))
            else:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()
        optimizer_noise.step()


def test(epoch):
    global acc, acc_best, epoch_best, acc5_best
    net.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = images.cuda(), labels.cuda()
            output, output_feature_s = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        torch.save(net.state_dict(), args.output_dir + 'student/' + 'checkpoint.pth')
        acc_best = acc
        epoch_best = epoch
    if acc5_best < top5:
        acc5_best = top5
    print('Test Avg. Loss: %f, Accuracy: %f. Epoch: %d' % (avg_loss.data.item(), acc, epoch))
    print('******** ******** ********')
    print('Test Avg Best. Accuracy: %f. Accuracy-5: %f. Epoch: %d' % (acc_best, acc5_best, epoch_best))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))


def train_and_test(epoch, trainloader3, nll, class_weight):
    train(epoch, trainloader3, nll, class_weight)
    test(epoch)

# def adjust_learning_rate(optimizer, epoch, max_epoch):
#     """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
#     if epoch < (max_epoch/200.0*80.0):
#         lr = 0.1
#     elif epoch < (max_epoch/200.0*160.0):
#         lr = 0.01
#     else:
#         lr = 0.001
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, max_epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr_S
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_positive(value_blk3, value_blk7, value_out, args):
    positive_index = []
    if args.attnprobe_sel:
        value = value_out
    else:
        value = args.attnlier * (value_blk3 + value_blk7) + args.outlier * value_out
    if args.random:
        print('randomly selected!')
        positive_index = torch.tensor(random.sample(range(1281167), args.num_select))
    else:
        positive_index = value.topk(args.num_select,largest=False)[1]
    return positive_index  


def main():
    global acc_best
    if os.path.exists(args.selected_file + 'value_out.pth'):
        value_out = torch.load(args.selected_file + 'value_out.pth').cuda()
        pred_out = torch.load(args.selected_file + 'pred_out.pth').cuda()
        value_blk3 = torch.load(args.selected_file + 'value_blk3.pth').cuda()
        value_blk7 = torch.load(args.selected_file + 'value_blk7.pth').cuda()
        # value_numpy = np.loadtxt(args.selected_file + '/value.txt')
        # value = torch.Tensor(value_numpy)
        # pred_numpy = np.loadtxt(args.selected_file + '/pred.txt')
        # pred = torch.Tensor(pred_numpy)
    else:
        value_blk3, value_blk7 = identify_attnlier(checkpoint, embed_dim, num_heads)
        value_out, pred_out = identify_outlier()
        torch.save(value_out, args.selected_file + 'value_out.pth')
        torch.save(pred_out, args.selected_file + 'pred_out.pth')
        torch.save(value_blk3, args.selected_file + 'value_blk3.pth')
        torch.save(value_blk7, args.selected_file + 'value_blk7.pth')
        # np.savetxt(args.selected_file + '/value.txt', value.numpy(), fmt='%d',delimiter=None) 
        # np.savetxt(args.selected_file + '/pred.txt', pred.numpy(), fmt='%d',delimiter=None) 
    positive_index = get_positive(value_blk3, value_blk7, value_out, args)
    nll = torch.nn.NLLLoss().cuda()
    positive_index = positive_index.tolist()

    data_train_select = torch.utils.data.Subset(data_train_transform, positive_index)
    trainloader3 = torch.utils.data.DataLoader(data_train_select, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    class_weight = get_class_weight(teacher, trainloader3, num_classes=args.nb_classes)
    print(class_weight)
    class_weights = perturb(class_weight)
    epoch = int(320000/args.num_select * 512)

    for e in range(1, epoch):
        adjust_learning_rate(optimizer, e, epoch, args)
        train_and_test(e, trainloader3, nll, class_weights)
    print(acc_best)


if __name__ == '__main__':
    main()
        
    
