#!/bin/bash
# original dataset: cifar, wild dataset: imagenet
root_cifar10='/database/cifar10/'
root_cifar100='/database/cifar100/'
root_wild='/database/imagenet/train/'
selected_cifar10='/selected/cifar10/'
selected_cifar100='/selected/cifar100/'
teacher_cifar10='/output/cifar10/teacher/checkpoint.pth'
teacher_cifar100='/output/cifar100/teacher/checkpoint.pth'
output_student_cifar10='/output/cifar10/'
output_student_cifar100='/output/cifar100/'
# CIFAR10
CUDA_VISIBLE_DEVICES=0 python DFND_DeiT-train.py --dataset cifar10 --data_cifar $root_cifar10 --data_imagenet $root_wild --num_select 650000 --teacher_dir $teacher_cifar10 --selected_file $selected_cifar10 --output_dir $output_student_cifar10 --nb_classes 10 --lr_S 7.5e-4 --attnprobe_sel --attnprobe_dist 
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python DFND_DeiT-train.py --dataset cifar100 --data_cifar $root_cifar100 --data_imagenet $root_wild --num_select 650000 --teacher_dir $teacher_cifar100 --selected_file $selected_cifar100 --output_dir $output_student_cifar100 --nb_classes 100 --lr_S 8.5e-4 --attnprobe_sel --attnprobe_dist 

