#!/bin/bash

# original dataset: imagenet, wild dataset: flicker1m
root_imagenet='/database/imagenet/'
root_wild='/database/flicker1m/train/'
selected_imagenet='/selected/imagenet/'
teacher_imagenet='/output/imagenet/teacher/deit_base_patch16_224-b5f2ef4d.pth'
output_student_imagenet='/output/imagenet/'
# imagenet
CUDA_VISIBLE_DEVICES=0 python DFND_DeiT-imagenet.py --dataset imagenet --data_cifar $root_imagenet --data_imagenet $root_wild --num_select 1000000 --teacher deit_base_patch16_224 --teacher_dir $teacher_imagenet --selected_file $selected_imagenet --output_dir $output_student_imagenet --nb_classes 1000 --pos_num 129 --lr_S 7.5e-4 --attnprobe_sel --attnprobe_dist

