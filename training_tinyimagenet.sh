#!/bin/bash

# original dataset: tinyimagenet, wild dataset: flicker1m
root_tinyimagenet='/database/tiny-imagenet/'
root_wild='/database/flicker1m/'
selected_tinyimagenet='/selected/tinyimagenet/'
teacher_tinyimagenet='/output/tinyimagenet/teacher/checkpoint.pth'
output_student_tinyimagenet='/output/tinyimagenet/'
# tinyimagenet
CUDA_VISIBLE_DEVICES=0 python DFND_DeiT-imagenet.py --dataset tinyimagenet --data_cifar $root_tinyimagenet --data_imagenet $root_wild --num_select 800000 --teacher deit_small_patch16_224 --teacher_dir $teacher_tinyimagenet --selected_file $selected_tinyimagenet --output_dir $output_student_tinyimagenet --nb_classes 200 --pos_num 50 --lr_S 7.5e-4 --attnprobe_sel --attnprobe_dist 
