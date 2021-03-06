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

def __main(data_path):
    from experiments import train_pytorch, test_pytorch
    train_pytorch.__main(data_path,default_config_file='charades_i3d_tc4_f1024.yaml')
    
    # from experiments import train_pytorch_batchMap
    # train_pytorch_batchMap.__main(default_config_file = 'charades_i3d_tc3_f256.yaml')
    
    # or using pytorch
    # test_pytorch.__main()

if __name__ == '__main__':
    # data_path = '/home/r/renpengzhen/Datasets/Charades' #原始数据集及提取的特征存放的路径
    data_path = '/data/renpengzhen/data/Charades' #原始数据集及提取的特征存放的路径
    __main(data_path)
