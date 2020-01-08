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
Train Timeception layers on different datasets. There are two different ways to train Timeception.
 1. Timeception-only (TCO): only timeception layers are trained, using features extracted from backbone CNNs.
 2. End-to-end (ETE): timeception is trained on top of backbone CNN. The input is video frames passed throughtout the backboneCNN
    and then the resulted feature is fed to Timeception layers. Here, you enjoy all the benefits of end-to-end training.
    For example, do pre-processing to the input frames, randomly sample the frames, temporal jittering, ...., etc.
串联帧进行特征提取
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import logging
import datetime
import numpy as np
from optparse import OptionParser

import torch
import torch.utils.data

from torch.nn import functional as F
from torch.nn import Module, Dropout, BatchNorm1d, LeakyReLU, Linear, LogSoftmax, Sigmoid
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader


from nets import timeception_pytorch
from core import utils, pytorch_utils, image_utils, config_utils, const, config, data_utils_pytorch, metrics
from core.utils import Path as Pth
# from tensorboardX import SummaryWriter as writer

logger = logging.getLogger(__name__)

from datasets import charades
def extract_feature_i3d_for_train(data_root, n_frames_in, concate_feature, random_sample):
    #为测试进行视频随机采样，并进行特征提取
    print('--- sampling frames')
    print('start time: %s' % utils.timestamp())
    annotation_path = charades._13_prepare_annotation_frames_per_video_dict_untrimmed_multi_label_for_i3d(n_frames_in, data_root, random_sample, is_training=True)
    charades.extract_features_i3d_charades(data_root, n_frames_in,annotation_path, concate_feature, random_sample, is_training=True)

def train_tco(data_path,concate_feature=False,random_sample = False):
    """
    Train Timeception layers based on the given configurations.
    This train scheme is Timeception-only (TCO).
    """

    # get some configs for the training
    n_timestep_in = config.cfg.MODEL.N_TC_TIMESTEPS
    n_epochs = config.cfg.TRAIN.N_EPOCHS #500
    dataset_name = config.cfg.DATASET_NAME #Charades
    model_name = '%s_%s_%s' % (config.cfg.MODEL.NAME, n_timestep_in, utils.timestamp())
    #'charades_timeception_19.08.05-10:59:25'
    device = 'cuda'
    torch.cuda.set_device(0)  # 设置当前设备
    
    print('--- start time')
    print(datetime.datetime.now())
    print('model_name: %s' % model_name)
    
    #生成特征
    data_root = '/'.join(data_path.split('/')[:-1])  # '/data/renpengzhen/data'
    n_frames_in = n_timestep_in * 8  # 需要的帧的个数
    # extract_feature_i3d_for_train(data_root, n_frames_in, concate_feature, False)  # 如果是随机采帧训练，第一个数据集采用等距离采样及合并特征提取

    # data generators 加载数据集
    loader_tr, n_samples_tr, n_batches_tr = __define_loader(data_path,is_training=True) #n_samples_tr = 7811，n_batches_tr=245
    loader_te, n_samples_te, n_batches_te = __define_loader(data_path,is_training=False)#n_samples_te=1814,n_batches_te=37
    print('... [tr]: n_samples, n_batch, batch_size: %d, %d, %d' % (n_samples_tr, n_batches_tr, config.cfg.TRAIN.BATCH_SIZE))
    print('... [te]: n_samples, n_batch, batch_size: %d, %d, %d' % (n_samples_te, n_batches_te, config.cfg.TEST.BATCH_SIZE))


    # load model，这里进行加载已经构建好的模型框架
    model, optimizer, loss_fn, metric_fn, metric_fn_name = __define_timeception_model(device)
    
    # 打印模型参数的大小，即所占空间
    print('param size = %f MB'%utils.count_parameters_in_MB(model))

    # print('batch_size=2, input_shape[1:]=', model._input_shape[1:])
    # print(pytorch_utils.summary(model, model._input_shape[1:], batch_size=2, device='cuda'))#打印模型摘要

    # save the model，保存模型状态
    model_saver = pytorch_utils.ModelSaver(model, dataset_name, model_name)


    # loop on the epochs
    for idx_epoch in range(n_epochs):
        epoch_num = idx_epoch + 1
        # print(epoch_num)
        loss_tr = 0.0
        loss_te = 0.0
        tt1 = time.time()

        # flag model as training
        model.train() #将模型设置为训练阶段
        # training
        Y_true, Y_pred = np.empty([0, 157]), np.empty([0, 157])
        duration = 0.0
        loss_b_tr = 0.0
        for idx_batch, (x, y_true) in enumerate(loader_tr):
            batch_num = idx_batch + 1
    
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
    
            # calculate accuracy
            y_true = y_true.cpu().numpy().astype(np.int32)
            y_pred = y_pred.cpu().detach().numpy()
            Y_pred = np.append(Y_pred, y_pred, axis=0)
            Y_true = np.append(Y_true, y_true, axis=0)
            loss_b_tr = loss.cpu().detach().numpy()
    
            loss_tr += loss_b_tr
            loss_b_tr = loss_tr / float(batch_num)
            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds - epoch: %02d/%02d, batch [tr]: %02d/%02d, loss: %0.4f' % (duration, epoch_num, n_epochs, batch_num, n_batches_tr, loss_b_tr))
        acc_tr = metric_fn(Y_true, Y_pred)  # 训练完一个epoch上的准确率
        loss_tr /= float(n_batches_tr)
        sys.stdout.write('\r%04ds - epoch: %02d/%02d, loss: %0.4f, map: %0.4f' % (duration, epoch_num, n_epochs, loss_b_tr, acc_tr))
        # after each epoch, save data
        model_saver.save(idx_epoch)

        # flag model as testing
        model.eval()
        # testing
        Y_true, Y_pred = np.empty([0, 157]), np.empty([0, 157])
        for idx_batch, (x, y_true) in enumerate(loader_te):
            batch_num = idx_batch + 1
    
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            loss_b_te = loss_fn(y_pred, y_true).cpu().detach().numpy()
            y_true = y_true.cpu().numpy().astype(np.int32)
            y_pred = y_pred.cpu().detach().numpy()
    
            loss_te += loss_b_te
            loss_b_te = loss_te / float(batch_num)
            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds - epoch: %02d/%02d, batch [te]: %02d/%02d, loss, %s: %0.2f' % (duration, epoch_num, n_epochs, batch_num, n_batches_te, metric_fn_name, loss_b_te))
            Y_pred = np.append(Y_pred, y_pred, axis=0)
            Y_true = np.append(Y_true, y_true, axis=0)
        
        acc_te = metric_fn(Y_true, Y_pred)
        loss_te /= float(n_batches_te)

        tt2 = time.time()
        duration = tt2 - tt1
        sys.stdout.write('\r%04ds - epoch: %02d/%02d, [tr]: %0.4f, %0.4f, [te]: %0.4f, %0.4f\n' % (duration, epoch_num, n_epochs, loss_tr, acc_tr, loss_te, acc_te))
        
        # 重新生成数据
        if random_sample:
            if epoch_num < n_epochs:
                extract_feature_i3d_for_train(data_root, n_frames_in, concate_feature, random_sample)  # 包含随机采帧及特征提取
                loader_tr, n_samples_tr, n_batches_tr = __define_loader(data_path,is_training=True)  # 只重新加载训练集，测试集暂且保持不变

    print('--- finish time')
    print(datetime.datetime.now())

def train_ete():
    """
    Train Timeception layers based on the given configurations.
    This train scheme is End-to-end (ETE).
    """

    raise Exception('Sorry, not implemented yet!')

def __define_loader(data_path,is_training):
    """
    Define data loader.
    """

    # get some configs for the training，配置数据加载的参数
    #这里的参数以'/data/pengzhen/timeception-master/configs/charades_i3d_tc3_f256.yaml'这个配置文件中的信息为准
    n_classes = config.cfg.MODEL.N_CLASSES #157
    dataset_name = config.cfg.DATASET_NAME #charades
    backbone_model_name = config.cfg.MODEL.BACKBONE_CNN #i3d_pytorch_charades_rgb
    backbone_feature_name = config.cfg.MODEL.BACKBONE_FEATURE #mixed_5c
    n_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS #32
    n_workers = config.cfg.TRAIN.N_WORKERS #读取数据的线程数

    batch_size_tr = config.cfg.TRAIN.BATCH_SIZE #32
    batch_size_te = config.cfg.TEST.BATCH_SIZE #64
    batch_size = batch_size_tr if is_training else batch_size_te

    # size and name of feature
    feature_name = 'features_%s_%s_%sf' % (backbone_model_name, backbone_feature_name, n_timesteps) #'features _i3d_pytorch_charades_rgb_ mixed_5c_32f'
    c, h, w = utils.get_model_feat_maps_info(backbone_model_name, backbone_feature_name)
#features_i3d_pytorch_charades_rgb,mixed_5c,||获得模型对应的feature_map的大小细节：c,h,w = 1024, 7, 7
    feature_dim = (c, n_timesteps, h, w) #特征的维度：1024, 32, 7, 7，其中的n_timesteps是自己设定的

    # data generators
    params = {'batch_size': batch_size, 'n_classes': n_classes, 'feature_name': feature_name, 'feature_dim': feature_dim, 'is_training': is_training,'data_path': data_path}
    dataset_class = data_utils_pytorch.PYTORCH_DATASETS_DICT[dataset_name] #core.data_utils_pytorch.DatasetCharades
    dataset = dataset_class(**params)
    n_samples = dataset.n_samples #7811 1814
    n_batches = dataset.n_batches #245 37

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)

    return data_loader, n_samples, n_batches

def __define_timeception_model(device):
    """
    Define model, optimizer, loss function and metric function.
    """
    # some configurations一些参数配置
    classification_type = config.cfg.MODEL.CLASSIFICATION_TYPE #'ml'
    solver_name = config.cfg.SOLVER.NAME #'adam'
    solver_lr = config.cfg.SOLVER.LR #0.01
    adam_epsilon = config.cfg.SOLVER.ADAM_EPSILON #0.0001

    # define model
    model = Model().to(device)
    model_param = model.parameters()


    # define the optimizer
    optimizer = SGD(model_param, lr=0.01) if solver_name == 'sgd' else Adam(model_param, lr=solver_lr, eps=adam_epsilon)

    # loss and evaluation function for either multi-label "ml" or single-label "sl" classification
    if classification_type == 'ml':
        loss_fn = torch.nn.BCELoss()
        metric_fn = metrics.map_charades
        metric_fn_name = 'map'
    else:
        loss_fn = torch.nn.NLLLoss()
        metric_fn = metrics.accuracy
        metric_fn_name = 'acc'

    return model, optimizer, loss_fn, metric_fn, metric_fn_name

class Model(Module):
    """
    Define Timeception classifier.
    """

    def __init__(self):
        super(Model, self).__init__()

        # some configurations for the model
        n_tc_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS #32
        backbone_name = config.cfg.MODEL.BACKBONE_CNN #'i3d_pytorch_charades_rgb'
        feature_name = config.cfg.MODEL.BACKBONE_FEATURE #'mixed_5c'
        n_tc_layers = config.cfg.MODEL.N_TC_LAYERS #2
        n_classes = config.cfg.MODEL.N_CLASSES #157
        is_dilated = config.cfg.MODEL.MULTISCALE_TYPE #采用的多核策略类型'ks'
        OutputActivation = Sigmoid if config.cfg.MODEL.CLASSIFICATION_TYPE == 'ml' else LogSoftmax
        n_channels_in, channel_h, channel_w = utils.get_model_feat_maps_info(backbone_name, feature_name) #1024, 7, 7
        n_groups = int(n_channels_in / 128.0) #8

        input_shape = (None, n_channels_in, n_tc_timesteps, channel_h, channel_w)  # (None, C, T, H, W),(None, 1024, 32, 7, 7)其中T是自己设定的
        self._input_shape = input_shape #(None, 1024, 32, 7, 7)

        # define 4 layers of timeception
        self.timeception = timeception_pytorch.Timeception(input_shape, n_tc_layers, n_groups, is_dilated)

        # get number of output channels after timeception
        n_channels_in = self.timeception.n_channels_out

        # define layers for classifier
        self.do1 = Dropout(0.5)
        self.l1 = Linear(n_channels_in, 512)
        self.bn1 = BatchNorm1d(512)
        self.ac1 = LeakyReLU(0.2)
        self.do2 = Dropout(0.25)
        self.l2 = Linear(512, n_classes)
        self.ac2 = OutputActivation()

    def forward(self, input):
        # feedforward the input to the timeception layers
        tensor = self.timeception(input)
        
        # max-pool over space-time
        bn, c, t, h, w = tensor.size()
        tensor = tensor.view(bn, c, t * h * w)
        tensor = torch.max(tensor, dim=2, keepdim=False) #size: bn * c
        tensor = tensor[0] #把value给tensor

        # dense layers for classification
        tensor = self.do1(tensor)
        tensor = self.l1(tensor)
        tensor = self.bn1(tensor)
        tensor = self.ac1(tensor)
        tensor = self.do2(tensor)
        tensor = self.l2(tensor)
        tensor = self.ac2(tensor)

        return tensor

def __main(data_path, default_config_file,concate_feature,random_sample):
    """
    Run this script to train Timeception.
    """

    # Parse the arguments
    parser = OptionParser() #创建OptionParser对象，用于设置参数配置文件
    #使用parser.add_option(...)待定义命令行参数，及其帮助文档
    parser.add_option('-c', '--config_file', dest='config_file', default=default_config_file, help='Yaml config file that contains all training details.')
    (options, args) = parser.parse_args()
    #option: {'config_file': 'charades_i3d_tc2_f256.yaml'},args: []
    #options 是一个字典，其key字典中的关键字可能会是我们所有的add_option()函数中的dest参数值，其对应的value值，是命令行输入的对应的add_option()函数的参数值。
    #args,它是一个由 positional arguments 组成的列表。
    config_file = options.config_file #'charades_i3d_tc2_f256.yaml'
    

    # check if exist,不存在进行警告
    if config_file is None or config_file == '':
        msg = 'Config file not passed, default config is used: %s' % (config_file)
        logging.warning(msg)
        config_file = default_config_file

    # path of config file
    config_path = './configs/%s' % (config_file) #'./configs/charades_i3d_tc2_f256.yaml'

    # check if file exist不存在进行警告
    if not os.path.exists(config_path):
        msg = 'Sorry, could not find config file with the following path: %s' % (config_path)
        logging.error(msg)

    else:
        # read the config from file and copy it to the project configuration "cfg"，从文件中读取配置并将其复制到项目配置“config。py”中
        config_utils.cfg_from_file(config_path)

        # choose which training scheme, either 'ete' or 'tco'
        training_scheme = config.cfg.TRAIN.SCHEME #tco

        # start training
        if training_scheme == 'tco':
            train_tco(data_path,concate_feature,random_sample)
        else:
            train_ete()

