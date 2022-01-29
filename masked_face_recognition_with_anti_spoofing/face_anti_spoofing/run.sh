#!/bin/bash

# ./run.sh > log.txt 2>&1 & 
python main.py --model=ResNet18 --modality=color  --patch_size=48 --epoch_inter=10 --cycle_num=1 --initial_lr=0.001 \
									  --pretrained_model=global_min_acer_model1.pth
									  #--attention=True #--bottleneck=True #只有True时才pass in，否则comment掉 \
									  