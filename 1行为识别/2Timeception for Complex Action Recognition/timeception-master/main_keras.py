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

"""
Main file of the project.
"""
import os
import pickle
def py3_pkl_py2():
    # python3生成的pkl文件向python2进行转换
    root_path = '/home/r/renpengzhen/PyTorch/NAS_AR_CNN/data/Charades'
    feature_name = 'features_i3d_pytorch_charades_rgb_mixed_5c_32f'
    feats_path = os.path.join(root_path,feature_name)
    transform_path = '%s/%s_python2'%(root_path,feature_name)
    if not os.path.exists(transform_path):
        os.makedirs(transform_path)
    count = 0
    for root,dirs,files in os.walk(feats_path):
        print(len(files))
        for file in files:
            count += 1
            if count%100==0:
                print(count)
            file_path = os.path.join(root,file)
            save_path = os.path.join(transform_path,file)
            with open(file_path, 'rb') as f:
                w = pickle.load(f)
                pickle.dump(w, open(save_path, "wb"), protocol=2)

def __main():
    from experiments import train_keras, test_keras

    # to train Timeception using keras
    # default_config_file = 'charades_i3d_tc4_f1024.yaml'
    default_config_file = 'charades_i3d_tc3_f256.yaml'
    # train_keras.__main(default_config_file)


    # to test Timeception using keras
    test_keras.__main(default_config_file)

if __name__ == '__main__':
    # py3_pkl_py2()
    # exit()
    __main()
    
    pass
