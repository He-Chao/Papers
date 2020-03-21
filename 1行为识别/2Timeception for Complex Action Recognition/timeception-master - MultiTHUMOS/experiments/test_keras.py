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
Test Timeception models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import datetime
from optparse import OptionParser

import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LeakyReLU, Dropout, Input, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization

from nets import timeception
from nets.layers_keras import MaxLayer
from core import utils, keras_utils, image_utils, config_utils, const, config, data_utils_keras, metrics
from core.utils import Path as Pth
import os
from keras.models import model_from_json
from experiments.train_keras import __define_data_generator,__define_timeception_model
import numpy as np

logger = logging.getLogger(__name__)

def test_tco():
    model_path = '/home/r/renpengzhen/PyTorch/timeception-master/data/Charades/models/charades_timeception_19.10.25-09:47:34'
    n_epochs = len(os.listdir(model_path)) // 2
    i = 0
    model_file_path = []
    weight_file_path = []
    data_generator_te = __define_data_generator(is_training=False)
    for root, dirs, files in os.walk(model_path):
        files.sort()
        for file in files:
            if i % 2 == 0:
                model_file_path += [os.path.join(root, file)]
            else:
                weight_file_path += [os.path.join(root, file)]
            i += 1
    #导入模型结构
    # model_json_path = model_file_path[0]
    # with open(model_json_path, 'r') as file:
    #     model_json1 = file.read()
    #     model = model_from_json(model_json1)
    model = __define_timeception_model()
    for epoch in range(n_epochs):
        Y_true, Y_pred = np.empty([0, 157]), np.empty([0, 157])  # 构建一个空的np数组
        model_weight_path = weight_file_path[epoch]
        model.load_weights(model_weight_path)
        for i, (x_test, y_test) in enumerate(data_generator_te):
            y_pred = model.predict(x_test)
            Y_true = np.append(Y_true, y_test, axis=0)
            Y_pred = np.append(Y_pred, y_pred, axis=0)
        acc = metrics.map_charades(Y_true, Y_pred)
        logging.info(epoch, ' acc : ', acc)

def __main(default_config_file):
    # Parse the arguments
    parser = OptionParser()
    parser.add_option('-c', '--config_file', dest='config_file', default=default_config_file,
                      help='Yaml config file that contains all training details.')
    (options, args) = parser.parse_args()
    config_file = options.config_file
    
    # check if exist
    if config_file is None or config_file == '':
        msg = 'Config file not passed, default config is used: %s' % (config_file)
        logging.warning(msg)
        config_file = default_config_file
    
    # path of config file
    config_path = './configs/%s' % (config_file)
    # check if file exist
    if not os.path.exists(config_path):
        msg = 'Sorry, could not find config file with the following path: %s' % (config_path)
        logging.error(msg)
    else:
        # read the config from file and copy it to the project configuration "cfg"
        config_utils.cfg_from_file(config_path)
    
        # choose which training scheme, either 'ete' or 'tco'
        training_scheme = config.cfg.TRAIN.SCHEME
    
        # start training
        if training_scheme == 'tco':
            test_tco()
        else:
            train_ete()
    
    

if __name__ == '__main__':
    __main(default_config_file)