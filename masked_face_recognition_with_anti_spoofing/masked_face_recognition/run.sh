#!/bin/bash

# /bin/bash run.sh > log.txt 2>&1 & 
#只有True才写上，若是False则comment掉
# python main_softmax.py --model=resnet50 --epochs=10 --attention=True --cbam=True --masked_face=True\
# 						--pretrained_model=global_max_auc_model_3.pth #--mode=validate


# python main_triplet.py --model=resnet50 --epochs=1 --attention=True --cbam=True --masked_face=True\
# 						--pretrained_model=global_max_acc_model.pth
						#--fusion_face=True\
					   
python main_triplet_att.py --model=resnet50 --epochs=10 --cbam=True --masked_face=True\
						   --pretrained_model=global_max_auc_model_3.pth\
						     --fusion_face=True\