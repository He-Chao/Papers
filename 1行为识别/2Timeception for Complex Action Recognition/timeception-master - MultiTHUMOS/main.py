#!/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################
# PyTorch 1.0.1

"""
Main file of the project.
"""
from experiments import train_pytorch,test_pytorch,train_pytorch_concate_feature

def default_config(n_timestep_in):
    if n_timestep_in in [32,64]:
        cnn_n_timestep_in = n_timestep_in * 8
        default_config_file = 'MultiTHUMOS_i3d_tc3_f%d.yaml'%cnn_n_timestep_in
    else:
        cnn_n_timestep_in = n_timestep_in * 8
        default_config_file = 'MultiTHUMOS_i3d_tc4_f%d.yaml' % cnn_n_timestep_in
    return default_config_file

'''
    #训练阶段
    train_pytorch.__main(data_path,default_config_file)
    
    #串联帧一起提取特征进行训练
    train_pytorch_concate_feature.__main(data_path,default_config_file)
    
    
    #测试阶段
    model_weight_path = '/home/amax/renpengzhen/timeception-master/data/Charades/models/charades_timeception_19.12.22-04:05:46/013.pt' #测试时模型的权重
    test_pytorch.__main(data_path, default_config_file, model_weight_path, random_sample=True)
'''

if __name__ == '__main__':
    n_timestep_in = 128
    # data_path = '/home/r/renpengzhen/Datasets/Charades' #原始数据集及提取的特征存放的路径
    data_path = '/data/renpengzhen/data/MultiTHUMOS/i3d_feature' #原始数据集及提取的特征存放的路径
    
    # data_path = '/media/amax/RPZ/Datasets/MultiTHUMOS/i3d_feature_con' #原始数据集及提取的特征存放的路径
    default_config_file = default_config(n_timestep_in)
    train_pytorch_concate_feature.__main(data_path, default_config_file, concate_feature=False, random_sample=False) # 串联帧一起提取特征进行训练
    exit()

    
    # 测试：
    model_weight_path = '/home/amax/renpengzhen/timeception-master/data/Charades/models/charades_timeception_32_19.12.23-06:38:49/017-32.89.pt' #测试时模型的权重
    test_pytorch.__main(data_path, default_config_file, model_weight_path, n_epochs = 50,concate_feature=True) #这里测试必须使用随机采样，不然没有那么多的数据
    exit()
    
    # 串联帧一起提取特征进行训练
    train_pytorch_concate_feature.__main(data_path, default_config_file, concate_feature=True,random_sample = False)
    exit()
    
    # 对训练集进行随机采样，测试集保持不边，合并特征提取
    train_pytorch_concate_feature.__main(data_path, default_config_file, concate_feature=True,random_sample = True)


