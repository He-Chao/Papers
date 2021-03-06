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
Some stats for the charades dataset:
No. of sample / no. of frames / avg. frames per sample

train:  49809   15,161,387    304.4
test :  16691   4,817,413     288.6

when sampling 20 frames per action sample (i.e. instance), that what we get
because we have 7 training samples have no frames
49802
16691
total: 66493

For KineticsMicro
5750
1250
total: 7000

We have in total 9848 videos:
videos/ videos with no action
train: 7985 / 7811
test : 1863 / 1814
total: 9848 / 9625

Frames per video
-----------------------
# avg, min, max
# 663.746895404
# 0
# 6515

# 729.507166483
# 55
# 7983
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from optparse import OptionParser

import numpy as np
import torch
import csv
import os
import sys
import shutil
# import parse
import time
import threading
import cv2
# import imageio
from natsort import natsort
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import torchsummary
import logging

# from core import charades_utils
from nets import i3d_torch_charades_utils
from core import const as c
from core import utils, image_utils, config_utils

logger = logging.getLogger(__name__)

# region Constants

N_CLASSES = 157
N_VIDEOS = 9848
N_FRAMES_PER_SECOND = 24

# endregion

# region Prepare Annotation

def _01_prepare_annotation_class_names():
    root_path = c.data_root_path
    annot_text_path = '%s/Charades/annotation/Charades_v1_classes.txt' % (root_path)
    annot_pkl_path = '%s/Charades/annotation/class_names.pkl' % (root_path)

    class_names = utils.txt_load(annot_text_path)

    class_ids = [int(n[1:5]) for n in class_names]
    for i_1, i_2 in zip(class_ids, np.arange(N_CLASSES)):
        assert i_1 == i_2

    class_names = [n[5:] for n in class_names]
    class_names = np.array(class_names)

    utils.pkl_dump(class_names, annot_pkl_path, is_highest=True)
    _ = 10

def _02_prepare_annotation_frame_dict(is_training=True):
    root_path = c.data_root_path
    annot_tr_text_path = '%s/Charades/annotation/Charades_v1_train.csv' % (root_path)
    annot_te_text_path = '%s/Charades/annotation/Charades_v1_test.csv' % (root_path)
    annotation_pkl_tr_path = '%s/Charades/annotation/frames_dict_tr.pkl' % (root_path)
    annotation_pkl_te_path = '%s/Charades/annotation/frames_dict_te.pkl' % (root_path)

    annot_text_path = annot_tr_text_path if is_training else annot_te_text_path
    annotation_pkl_path = annotation_pkl_tr_path if is_training else annotation_pkl_te_path
    annotation_dict = {}
    n_actions = N_CLASSES

    frames_per_instance = []

    # add empty list for each action in the annotation dictionary
    for idx_action in range(n_actions):
        action_num = idx_action + 1
        annotation_dict[action_num] = []

    with open(annot_text_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            action_strings = row['actions']
            action_strings_splits = action_strings.split(';')
            video_id = row['id']
            if len(action_strings) == 0:
                print('... no action for video %s' % (video_id))
                continue
            for action_st in action_strings_splits:
                action_splits = action_st.split(' ')
                action_idx = int(action_splits[0][1:])
                action_num = action_idx + 1
                action_start = action_splits[1]
                action_end = action_splits[2]

                # add frames
                frames_relative_path = __get_frames_relative_pathes_in_given_duration(video_id, action_start, action_end)
                annotation_dict[action_num].append(frames_relative_path)

                # accumulate counter
                n_frames_per_instance = len(frames_relative_path)
                frames_per_instance.append(n_frames_per_instance)

    # save annotation
    utils.pkl_dump(annotation_dict, annotation_pkl_path, is_highest=True)
    print(frames_per_instance)
    print(len(frames_per_instance))
    print(np.sum(frames_per_instance))
    print(np.average(frames_per_instance))

def _03_prepare_annotation_frame_list():
    """
    Convert the annotation dict to list. Also, create list for ground truth.
    """

    n_frames_per_sample = 20
    n_classes = N_CLASSES

    root_path = c.data_root_path
    annotation_dict_tr_path = '%s/Charades/annotation/frames_dict_tr.pkl' % (root_path)
    annotation_dict_te_path = '%s/Charades/annotation/frames_dict_te.pkl' % (root_path)
    annotation_list_path = '%s/Charades/annotation/frames_list_%d_frames.pkl' % (root_path, n_frames_per_sample)

    annotation_dict_tr = utils.pkl_load(annotation_dict_tr_path)
    annotation_dict_te = utils.pkl_load(annotation_dict_te_path)

    x_tr = []
    x_te = []
    y_tr = []
    y_te = []

    class_nums = range(1, n_classes + 1)
    for class_num in class_nums:
        print('... %d/%d' % (class_num, n_classes))
        class_annot_tr = annotation_dict_tr[class_num]
        class_annot_te = annotation_dict_te[class_num]

        for sample_tr in class_annot_tr:
            n_f = len(sample_tr)
            if n_f == 0:
                print('zero frames in tr sample')
                continue
            if n_f < n_frames_per_sample:
                idx = np.random.randint(low=0, high=n_f, size=(n_frames_per_sample,))
            else:
                idx = np.random.choice(n_f, n_frames_per_sample)
            sample_frame_pathes = np.array(sample_tr)[idx]
            x_tr.append(sample_frame_pathes)
            y_tr.append(class_num)

        for sample_te in class_annot_te:
            n_f = len(sample_te)
            if n_f == 0:
                print('zero frames in te sample')
                continue
            if n_f < n_frames_per_sample:
                idx = np.random.randint(low=0, high=n_f, size=(n_frames_per_sample,))
            else:
                idx = np.random.choice(n_f, n_frames_per_sample)
            sample_frame_pathes = np.array(sample_te)[idx]
            x_te.append(sample_frame_pathes)
            y_te.append(class_num)

    x_tr = np.array(x_tr)
    x_te = np.array(x_te)
    y_tr = np.array(y_tr)
    y_te = np.array(y_te)

    print(x_tr.shape)
    print(y_tr.shape)
    print(x_te.shape)
    print(y_te.shape)
    data = (x_tr, y_tr, x_te, y_te)

    utils.pkl_dump(data, annotation_list_path, is_highest=True)

def _06_prepare_video_annotation_multi_label():
    root_path = '.'
    video_annotation_path = '%s/Charades/annotation/video_annotation.pkl' % (root_path)
    video_annotation_multi_label_path = '%s/Charades/annotation/video_annotation_multi_label.pkl' % (root_path)

    (video_id_tr, y_tr, video_id_te, y_te) = utils.pkl_load(video_annotation_path)

    video_ids_tr = np.unique(video_id_tr)
    video_ids_te = np.unique(video_id_te)

    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)
    n_classes = N_CLASSES

    video_gt_dict_tr = dict()
    video_gt_dict_te = dict()

    for id in video_ids_tr:
        video_gt_dict_tr[id] = []

    for id in video_ids_te:
        video_gt_dict_te[id] = []

    for i, j in zip(video_id_tr, y_tr):
        video_gt_dict_tr[i].append(j)

    for i, j in zip(video_id_te, y_te):
        video_gt_dict_te[i].append(j)

    # binarize labels of videos
    y_multi_label_tr = np.zeros((n_tr, n_classes), dtype=np.int)
    y_multi_label_te = np.zeros((n_te, n_classes), dtype=np.int)

    for idx_video, video_name in enumerate(video_ids_tr):
        idx_class = np.add(video_gt_dict_tr[video_name], -1)
        y_multi_label_tr[idx_video][idx_class] = 1
        _ = 10

    for idx_video, video_name in enumerate(video_ids_te):
        idx_class = np.add(video_gt_dict_te[video_name], -1)
        y_multi_label_te[idx_video][idx_class] = 1
        _ = 10

    data = (video_ids_tr, y_multi_label_tr, video_ids_te, y_multi_label_te)
    utils.pkl_dump(data, video_annotation_multi_label_path)

def _08_prepare_annotation_frames_per_video_dict_multi_label():
    """
    Get list of frames from each video. With max 600 of each video and min 100 frames from each video.
    These frames will be used to extract features for each video.
    """

    min_frames_per_video = 100
    max_frames_per_video = 100

    root_path = c.data_root_path
    annot_tr_text_path = '%s/Charades/annotation/Charades_v1_train.csv' % (root_path)
    annot_te_text_path = '%s/Charades/annotation/Charades_v1_test.csv' % (root_path)
    annotation_path = '%s/Charades/annotation/frames_dict_multi_label.pkl' % (root_path)

    video_frames_dict_tr = __get_frame_names_from_csv_file(annot_tr_text_path, min_frames_per_video, max_frames_per_video)
    video_frames_dict_te = __get_frame_names_from_csv_file(annot_te_text_path, min_frames_per_video, max_frames_per_video)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), annotation_path, is_highest=True)

def _12_prepare_annotation_frames_per_video_dict_multi_label_all_frames():
    """
    Get list of frames from each video. All frames for each video.
    """

    n_frames_per_video = None
    root_path = c.data_root_path
    annot_tr_text_path = '%s/Charades/annotation/Charades_v1_train.csv' % (root_path)
    annot_te_text_path = '%s/Charades/annotation/Charades_v1_test.csv' % (root_path)
    annotation_path = '%s/Charades/annotation/frames_dict_multi_label_all_frames.pkl' % (root_path)

    video_frames_dict_tr = __get_frame_names_from_csv_file(annot_tr_text_path, n_frames_per_video, n_frames_per_video, sampling=False)
    video_frames_dict_te = __get_frame_names_from_csv_file(annot_te_text_path, n_frames_per_video, n_frames_per_video, sampling=False)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), annotation_path, is_highest=True)

def _13_prepare_annotation_frames_per_video_dict_untrimmed_multi_label_for_i3d(n_frames_per_video,root_Charades_path,random_sample = False,is_training=True):
    """
    为I3D模型从视频帧当中进行帧采样
    Uniformly sample sequences of frames form each video. Each sequences consists of 8 successive frames.
    n_frames_per_video = 1024 || 512 || 256
    """
 
    # root_path = c.DATA_ROOT_PATH
    root_path = './data'
    annot_tr_text_path = '%s/Charades/annotation/Charades_v1_train.csv' % (root_path) #'./data/Charades/annotation/Charades_v1_train.csv'
    annot_te_text_path = '%s/Charades/annotation/Charades_v1_test.csv' % (root_path)
    annotation_path = '%s/Charades/annotation/frames_dict_untrimmed_multi_label_i3d_%d_frames' % (root_path, n_frames_per_video)

    #进行采样：每8个连续帧作为一个视频段
    if is_training:
        # 训练阶段
        if random_sample:
            #在训练阶段如果随机采样，只对训练集进行随机采样，保持测试集不变
            annotation_path = '%s_train' % annotation_path
            video_frames_dict_tr = __get_frame_names_untrimmed_from_csv_file_for_i3d(annot_tr_text_path,n_frames_per_video,root_Charades_path, random_sample,is_training)
            video_frames_dict_te = dict()  # 只对训练集进行随机采样，保持测试集不变
        else:
            #在训练阶段如果不随机采样，将训练集和测试集都进行等距离采样
            video_frames_dict_tr = __get_frame_names_untrimmed_from_csv_file_for_i3d(annot_tr_text_path,n_frames_per_video,root_Charades_path, random_sample,is_training)
            video_frames_dict_te = __get_frame_names_untrimmed_from_csv_file_for_i3d(annot_te_text_path,n_frames_per_video,root_Charades_path, random_sample,is_training)
    else:
        # 测试阶段
        annotation_path = '%s_test' % annotation_path
        video_frames_dict_tr = dict()  # 训练集暂且不做随机采样
        video_frames_dict_te = __get_frame_names_untrimmed_from_csv_file_for_i3d(annot_te_text_path, n_frames_per_video,root_Charades_path, random_sample,is_training)
    annotation_path = '%s.pkl'%annotation_path
    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), annotation_path, is_highest=True)
    return annotation_path

def _14_prepare_annotation_frames_per_video_dict_untrimmed_multi_label_for_resnet_ordered():
    """
    Get list of frames from each video. With max 600 of each video and min 96 frames from each video.
    These frames will be used to extract features for each video.
    """

    # if required frames per video are 128, there are 51/6 out of 7986/1864 videos in training/testing splits that don't satisfy this
    # n_frames_per_video = 32 || 64 || 128
    n_frames_per_video = 32
    root_path = c.data_root_path
    annot_tr_text_path = '%s/Charades/annotation/Charades_v1_train.csv' % (root_path)
    annot_te_text_path = '%s/Charades/annotation/Charades_v1_test.csv' % (root_path)
    annotation_path = '%s/Charades/annotation/frames_dict_untrimmed_multi_label_resnet_ordered_%d_frames.pkl' % (root_path, n_frames_per_video)

    video_frames_dict_tr = __get_frame_names_untrimmed_from_csv_file_for_ordered(annot_tr_text_path, n_frames_per_video, is_resnet=True)
    video_frames_dict_te = __get_frame_names_untrimmed_from_csv_file_for_ordered(annot_te_text_path, n_frames_per_video, is_resnet=True)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), annotation_path, is_highest=True)

def __get_frame_names_from_csv_file(annot_text_path, min_frames_per_video, max_frames_per_video, sampling=True):
    root_path = c.data_root_path
    counts_before = []
    counts_after = []
    count = 0
    video_frames_dict = dict()

    with open(annot_text_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if count % 100 == 0:
                print('%d' % (count))
            count += 1

            action_strings = row['actions']
            video_id = row['id']

            if len(action_strings) == 0:
                continue

            frames_relative_root_path = 'Charades/frames/Charades_v1_rgb/%s' % (video_id)
            frames_root_path = '%s/%s' % (root_path, frames_relative_root_path)

            frame_names = utils.file_names(frames_root_path, is_nat_sort=True)
            n_frames = len(frame_names)
            counts_before.append(n_frames)

            if sampling:
                frame_names = __sample_frames(frame_names, min_frames_per_video, max_frames_per_video)

            n_frames = len(frame_names)
            counts_after.append(n_frames)

            # sample from these frames
            video_frames_dict[video_id] = frame_names

    counts = np.array(counts_before)
    print('counts before')
    print(np.min(counts))
    print(np.max(counts))
    print(np.average(counts))
    print(len(np.where(counts < min_frames_per_video)[0]))
    print(len(np.where(counts < max_frames_per_video)[0]))

    counts = np.array(counts_after)
    print('counts after')
    print(np.min(counts))
    print(np.max(counts))
    print(np.average(counts))
    print(len(np.where(counts < min_frames_per_video)[0]))
    print(len(np.where(counts < max_frames_per_video)[0]))

    return video_frames_dict

def __get_frame_names_untrimmed_from_csv_file_for_ordered(annot_text_path, n_frames_per_video, is_resnet=False):
    counts = []
    count = 0
    video_frames_dict = dict()

    root_path = c.data_root_path
    n_lines = len(open(annot_text_path).readlines())

    with open(annot_text_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if count % 100 == 0:
                print('... %d/%d' % (count, n_lines))
            count += 1

            action_strings = row['actions']

            # if not action in the current video
            if len(action_strings) == 0:
                continue

            video_id = row['id']
            frames_root_path = '%s/charades/frames/Charades_v1_rgb/%s' % (root_path, video_id)
            video_frame_names = utils.file_names(frames_root_path, nat_sorted=True)

            if is_resnet:
                video_frame_names = __sample_frames_ordered_for_resnet(video_frame_names, n_frames_per_video)
            else:
                video_frame_names = __sample_frames_ordered(video_frame_names, n_frames_per_video)
            n_frames = len(video_frame_names)
            assert n_frames == n_frames_per_video
            counts.append(n_frames)

            # sample from these frames
            video_frames_dict[video_id] = video_frame_names

    counts = np.array(counts)
    print('counts before')
    print(np.min(counts))
    print(np.max(counts))
    print(np.average(counts))
    print(len(np.where(counts < n_frames_per_video)[0]))

    return video_frames_dict

def __get_frame_names_untrimmed_from_csv_file_for_i3d(annot_text_path, n_frames_per_video,root_Charades_path,random_sample=False, is_training=True):
    '''
    #video_frames_dict_tr = __get_frame_names_untrimmed_from_csv_file_for_i3d(annot_tr_text_path, n_frames_per_video)
    :param annot_text_path: './data/Charades/annotation/Charades_v1_train.csv'
    :param n_frames_per_video: 256
    :return: 返回采样后的视频帧词典
    '''
    count = 0
    video_frames_dict = dict()

    n_lines = len(open(annot_text_path).readlines())

    with open(annot_text_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sys.stdout.write('\r... %d/%d' % (count, n_lines)) #打印处理的进度
            count += 1
            action_strings = row['actions'] #视频的动作时间标注
            video_id = row['id'] #视频的id名称

            # some videos don't contain action annotation
            if len(action_strings) == 0:
                continue

            # get all frames of the video，得到该真video_id视频真实图片所存放的位置
            frames_relative_root_path = '%s/Charades_v1_rgb/%s' % (root_Charades_path,video_id) #得到对应视频的所有帧图片
            # frames_root_path = '%s/%s' % (root_path, frames_relative_root_path)
            #得到视频中所有图片的名字，即视频帧的名字
            video_frame_names = utils.file_names(frames_relative_root_path, is_nat_sort=True)

            # sample from these frames,在视频中进行采样
            if is_training:
                if random_sample:
                    video_frame_names = __random_sample_frames_for_i3d(video_frame_names, n_frames_per_video)
                else:
                    video_frame_names = __sample_frames_for_i3d(video_frame_names, n_frames_per_video)
            else:
                video_frame_names = __random_sample_frames_for_i3d(video_frame_names, n_frames_per_video)
            n_frames = len(video_frame_names)
            assert n_frames == n_frames_per_video

            #将采样后的视频帧存到词典中
            video_frames_dict[video_id] = video_frame_names
        sys.stdout.write('\n')

    return video_frames_dict

def __random_sample_frames_for_i3d(frames, n_required):
    # i3d model accepts sequence of 8 frames
    n_frames = len(frames)  # 该视频的总帧数597
    segment_length = 8
    n_segments = int(n_required / segment_length)  # 需要分段的个数256/8=32，需要将帧数分为32段
    
    assert n_required % segment_length == 0
    assert n_frames > segment_length
    
    if n_frames < n_required:
        # 视频帧数小于需要的帧数256，512，1024
        idces_start = np.random.randint(0, n_frames - segment_length, (n_segments,))
        idces_start = np.sort(idces_start)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()
    elif n_frames == n_required:
        # 相等的时候
        idx = np.arange(n_required)
    else:
        # 大于的时候
        idces_start = np.random.randint(0, n_frames - segment_length, (n_segments,))
        idces_start = np.sort(idces_start)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()
    sampled_frames = np.array(frames)[idx]
    return sampled_frames

def __sample_frames(frames, n_min, n_max):
    n_f = len(frames)

    if n_f < n_min:
        idx = np.random.randint(low=0, high=n_f, size=(n_min,))
    else:
        if n_f < n_max:
            idx = np.random.randint(low=0, high=n_f, size=(n_max,))
        else:
            idx = np.random.choice(n_f, n_max)

    sampled_frames = np.array(frames)[idx]
    return sampled_frames

def __sample_frames_ordered(frames, n_required):
    # get n frames
    n_frames = len(frames)

    if n_frames < n_required:
        repeats = int(n_required / float(n_frames)) + 1
        idx = np.arange(0, n_frames).tolist()
        idx = idx * repeats
        idx = idx[:n_required]
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        start_idx = int((n_frames - n_required) / 2.0)
        stop_idx = start_idx + n_required
        idx = np.arange(start_idx, stop_idx)

    sampled_frames = np.array(frames)[idx]
    assert len(idx) == n_required
    assert len(sampled_frames) == n_required
    return sampled_frames

def __sample_frames_ordered_for_resnet(frames, n_required):
    # 为resnet进行采帧
    # get n frames from all over the video
    n_frames = len(frames)

    if n_frames < n_required:
        step = n_frames / float(n_required)
        idx = np.arange(0, n_frames, step, dtype=np.float32).astype(np.int32)
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        step = n_frames / float(n_required)
        idx = np.arange(0, n_frames, step, dtype=np.float32).astype(np.int32)

    sampled_frames = np.array(frames)[idx]
    assert len(idx) == n_required
    assert len(sampled_frames) == n_required
    return sampled_frames

def __sample_frames_for_i3d(frames, n_required):
    # i3d model accepts sequence of 8 frames
    n_frames = len(frames) #该视频的总帧数597
    segment_length = 8
    n_segments = int(n_required / segment_length) #需要分段的个数256/8=32，需要将帧数分为32段

    assert n_required % segment_length == 0
    assert n_frames > segment_length

    if n_frames < n_required:
        #视频帧数小于需要的帧数256，512，1024
        step = (n_frames - segment_length) / float(n_segments)
        idces_start = np.arange(0, n_frames - segment_length, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()
    elif n_frames == n_required:
        #相等的时候
        idx = np.arange(n_required)
    else:
        #大于的时候
        step = n_frames / float(n_segments)
        idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()

    sampled_frames = np.array(frames)[idx]
    return sampled_frames

def __count_time_in_each_video(is_training=True):
    root_path = c.data_root_path
    annot_tr_text_path = '%s/Charades/annotation/Charades_v1_train.csv' % (root_path)
    annot_te_text_path = '%s/Charades/annotation/Charades_v1_test.csv' % (root_path)

    annot_text_path = annot_tr_text_path if is_training else annot_te_text_path
    frames_per_instance = []
    frames_per_videos = []
    time_per_videos = []

    count = 0

    with open(annot_text_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if count % 100 == 0:
                print('%d' % (count))
            count += 1

            action_strings = row['actions']
            action_strings_splits = action_strings.split(';')
            video_id = row['id']

            if len(action_strings) == 0:
                print('... no action for video %s' % (video_id))
                continue

            frames_relative_root_path = 'Charades/frames/Charades_v1_rgb/%s' % (video_id)
            frames_root_path = '%s/%s' % (root_path, frames_relative_root_path)
            frame_names = utils.file_names(frames_root_path, nat_sorted=True)

            frames_per_video = 0
            time_per_video = 0

            for action_st in action_strings_splits:
                action_splits = action_st.split(' ')
                action_start = action_splits[1]
                action_end = action_splits[2]

                action_time = float(action_end) - float(action_start)
                time_per_video += action_time

                idx_start = __convert_seconds_to_frame_idx(action_start)
                idx_stop = __convert_seconds_to_frame_idx(action_end)
                frame_names = frame_names[idx_start: idx_stop + 1]

                n_frames_per_instance = len(frame_names)
                frames_per_video += n_frames_per_instance
                frames_per_instance.append(n_frames_per_instance)

            time_per_videos.append(time_per_video)
            frames_per_videos.append(frames_per_video)

    print(frames_per_instance)
    print(len(frames_per_instance))
    print(np.sum(frames_per_instance))
    print(np.average(frames_per_instance))
    print(np.min(frames_per_instance))
    print(np.max(frames_per_instance))

    print(frames_per_videos)
    print(len(frames_per_videos))
    print(np.sum(frames_per_videos))
    print(np.average(frames_per_videos))
    print(np.min(frames_per_videos))
    print(np.max(frames_per_videos))

    print(time_per_videos)
    print(len(time_per_videos))
    print(np.sum(time_per_videos))
    print(np.average(time_per_videos))
    print(np.min(time_per_videos))
    print(np.max(time_per_videos))

    print(count)

def __count_how_many_videos_per_class():
    root_path = c.data_root_path
    annotation_path = '%s/Charades/annotation/video_annotation.pkl' % (root_path)
    (video_id_tr, y_tr, video_id_te, y_te) = utils.pkl_load(annotation_path)
    n_classes = N_CLASSES

    counts_tr = []
    counts_te = []

    for i in range(n_classes):
        counts_tr.append(len(np.where(y_tr == i + 1)[0]))
        counts_te.append(len(np.where(y_te == i + 1)[0]))

    counts_tr = np.array(counts_tr)
    counts_te = np.array(counts_te)

    idx = np.argsort(counts_tr)[::-1]
    counts_tr = counts_tr[idx]
    counts_te = counts_te[idx]

    counts = np.array([counts_tr, counts_te])
    print(counts_tr)
    print(counts_te)
    utils.plot_multi(counts, title='Counts')

def __test_video_names_in_annotation_list():
    n_frames_per_sample = 20
    root_path = c.data_root_path
    annotation_path = '%s/Charades/annotation/frames_list_%d_frames.pkl' % (root_path, n_frames_per_sample)

    (x_tr, y_tr, x_te, y_te) = utils.pkl_load(annotation_path)
    x = np.vstack((x_tr, x_te))
    n_videos = len(x)

    for i, video_pathes in enumerate(x):
        if i % 100 == 0:
            print('%d/%d' % (i, n_videos))
        video_name = video_pathes[0].split('/')[3] + '-'
        for p in video_pathes:
            assert video_name in p

def __get_frames_relative_pathes_in_given_duration(video_id, start_time_in_sec, stop_time_in_sec):
    """
    For a given video_id with start and stop time in seconds, get the relative pathes of the related frames.
    """
    root_path = c.data_root_path
    frames_relative_root_path = 'Charades/frames/Charades_v1_rgb/%s' % (video_id)
    frames_root_path = '%s/%s' % (root_path, frames_relative_root_path)
    frame_names = utils.file_names(frames_root_path, nat_sorted=True)

    idx_start = __convert_seconds_to_frame_idx(start_time_in_sec)
    idx_stop = __convert_seconds_to_frame_idx(stop_time_in_sec)

    frame_names = frame_names[idx_start: idx_stop + 1]
    frame_relative_pathes = ['%s/%s' % (frames_relative_root_path, n) for n in frame_names]
    return frame_relative_pathes

def __get_frames_names_in_given_duration(video_id, start_time_in_sec, stop_time_in_sec):
    """
    For a given video_id with start and stop time in seconds, get the relative pathes of the related frames.
    """
    root_path = c.data_root_path
    frames_relative_root_path = 'Charades/frames/Charades_v1_rgb/%s' % (video_id)
    frames_root_path = '%s/%s' % (root_path, frames_relative_root_path)
    frame_names = utils.file_names(frames_root_path, nat_sorted=True)

    idx_start = __convert_seconds_to_frame_idx(start_time_in_sec)
    idx_stop = __convert_seconds_to_frame_idx(stop_time_in_sec)

    frame_names = frame_names[idx_start: idx_stop + 1]
    return frame_names

def __convert_seconds_to_frame_idx(time_in_sec):
    time_in_sec = float(time_in_sec)
    idx = time_in_sec * N_FRAMES_PER_SECOND
    idx = round(idx)
    idx = int(idx)
    return idx

# endregion

# region Extract Features

def extract_features_i3d_charades(root_Charades_path,n_frames_in,frames_annot_path,concate_feature=False,random_sample=False,is_training= True):
    """
    Extract features from i3d-model
    n_frames_in = 8 * n_frames_out
    n_frames_in =  1024,512,256
    n_frames_out = 128 , 64, 32
    """

    # n_frames_in = 1024
    n_frames_out = n_frames_in // 8

    root_path = './data'
    # frames_annot_path = '%s/Charades/annotation/frames_dict_untrimmed_multi_label_i3d_%d_frames.pkl' % (root_path, n_frames_in) #采样过之后的帧路径
    # model_path = '/home/r/renpengzhen/PyTorch/timeception-master/model/i3d_kinetics_model_rgb.pth' #模型存放的位置
    model_path = '%s/Charades/baseline_models/i3d/rgb_charades.pt' % (root_path)  # 模型存放的位置
    frames_root_path = '%s/Charades_v1_rgb' % (root_Charades_path) #所有视频帧存放的位置
    features_root_path = '%s/Charades/features_i3d_pytorch_charades_rgb_mixed_5c_%df' % (root_Charades_path,n_frames_out) #用来存放使用i3d进行特征提取的路径
    sys.stdout.write('\rfeatures_save_path:%s\n'%features_root_path)
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)
    video_frames_dict = dict() #构建视频帧空词典
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)
    video_names = list(video_frames_dict.keys()) #视频的名字
    n_videos = len(video_names) #总视频的个数
    del video_frames_dict_tr
    del video_frames_dict_te

    n_threads = 8 #线程数
    n_frames_per_segment = 8 #每个视频段的帧数，这8帧是连续的，在采样的时候就是连续的
    assert n_frames_per_segment * n_frames_out == n_frames_in

    if not os.path.exists(features_root_path):
        os.makedirs(features_root_path)

    print('extracting training features')
    print('start time: %s' % utils.timestamp())

    # reader for getting video frames 用于获取视频帧的阅读器
    video_reader_tr = image_utils.AsyncVideoReaderCharadesForI3DTorchModel(n_threads=n_threads)

    # aync reader, and get load images for the first video, we will read the first group of videos
    video_group_frames = __get_video_frame_pathes(video_names[0], frames_root_path, video_frames_dict) #存储第一个视频帧的所有地址，是一个np数组类型
    video_reader_tr.load_video_frames_in_batch(video_group_frames)

    # load the model
    model = i3d_torch_charades_utils.load_model_i3d_charades_rgb_for_testing(model_path)

    # loop on list of videos，对整个视频数据集进行操作
    t0 = time.time()
    for idx_video in range(n_videos):
        video_num = idx_video + 1
        video_name = video_names[idx_video]

        # path to save the features，保存特征
        video_features_path = '%s/%s.pkl' % (features_root_path, video_name)  # 即将保存特征的路径
        if is_training:
            if random_sample:
                pass
            else:
                pass
                #如果不是随机采样，就判断文件中是否已经存在该特征，存在就不再重复提取
                #如果是随机采样的话，需要对特征进行覆盖
                # if os.path.exists(video_features_path):
                #     print('... features for video already exist: %s.pkl' % (video_name))
                #     continue
        
        begin_num = 0
        end_num = n_videos

        if begin_num is not None and end_num is not None:
            if video_num <= begin_num or video_num > end_num:
                continue

        # wait until the image_batch is loaded
        while video_reader_tr.is_busy():
            time.sleep(0.1)
        t1 = time.time()
        duration_waited = t1 - t0
        sys.stdout.write('\r... video %04d/%04d, waited: %.02f' % (video_num, n_videos, duration_waited))
        # get the frames
        frames = video_reader_tr.get_images()  # (G*T*N, 224, 224, 3)，这个我觉得是第一个视频里面裁剪过之后的帧图片

        # pre-load for the next video group, notice that we take into account the number of instances
        if video_num < n_videos:
            next_video_frames = __get_video_frame_pathes(video_names[idx_video + 1], frames_root_path, video_frames_dict)
            video_reader_tr.load_video_frames_in_batch(next_video_frames)

        if len(frames) != n_frames_in:
            raise ('... ... wrong n frames: %s' % (video_name))

        if concate_feature:
            frames = np.reshape(frames, (1, n_frames_in, 224, 224, 3))  #对采样的帧一起进行特征提取, (1, T*8, 224, 224, 3)，T实际上就是视频段，即超级帧的个数
        else:
            frames = np.reshape(frames, (n_frames_out, n_frames_per_segment, 224, 224, 3))  # (T, 8, 224, 224, 3)，T实际上就是视频段，即超级帧的个数
        # transpose to have the channel_first (T, 8, 224, 224, 3) => (T, 3, 8, 224, 224)
        frames = np.transpose(frames, (0, 4, 1, 2, 3))
        # prepare input variable
        with torch.no_grad():
            # extract features
            input_var = torch.from_numpy(frames).cuda()  # (T, 3, 8, 224, 224)，T=128,64,32
            output_var = model(input_var)  # 提取特征 torch.Size([128, 1024, 1, 7, 7])
            output_var = output_var.cpu()
            features = output_var.data.numpy()  # (T, 1024, 1, 7, 7)
            # don't forget to clean up variables
            del input_var
            del output_var
        if concate_feature:
            features = np.transpose(features, (2, 0, 3, 4, 1))  # (T, 1, 7, 7, 1024)
        else:
            features = np.transpose(features, (0, 2, 3, 4, 1))  # (T, 1, 7, 7, 1024)
        # reshape to have the features for each video in a separate dimension
        features = np.squeeze(features, axis=1)  # (T, 7, 7, 1024)，T=128,64,32
        # save features
        utils.pkl_dump(features, video_features_path, is_highest=True)

    t2 = time.time()
    print('\n... finish extracting features in %d seconds' % (t2 - t0))

def __relative_to_absolute_pathes(relative_pathes):
    # change relative to absolute pathes
    root_path = c.data_root_path
    absolute_pathes = np.array(['%s/%s' % (root_path, x) for x in relative_pathes])
    return absolute_pathes

def __get_video_frame_pathes(video_name, video_frames_root_path, video_frames_dict):
    video_frame_names = video_frames_dict[video_name] #获取视频帧的名字，是一个list
    video_frame_pathes = [('%s/%s/%s') % (video_frames_root_path, video_name, n) for n in video_frame_names] #视频帧路径，也是一个list
    video_frame_pathes = np.array(video_frame_pathes) #转换为np数组类型
    return video_frame_pathes

def __preprocess_img(img_path):
    # load image
    img = cv2.imread(img_path)

    img = image_utils.resize_crop(img)

    # as float
    img = img.astype(np.float32)

    # divide by 225 as caffe expect images to be in range 0-1
    img /= float(255)

    # also, swap to get RGB, as caffe expect RGB images
    img = img[:, :, (2, 1, 0)]

    return img

def __pre_process_for_charades(img):
    __img_mean = [0.485, 0.456, 0.406]
    __img_std = [0.229, 0.224, 0.225]
    img = image_utils.resize_crop(img)
    img = img.astype(np.float32)
    img /= float(255)
    img = img[:, :, (2, 1, 0)]
    img[:, :, 0] = (img[:, :, 0] - __img_mean[0]) / __img_std[0]
    img[:, :, 1] = (img[:, :, 1] - __img_mean[1]) / __img_std[1]
    img[:, :, 2] = (img[:, :, 2] - __img_mean[2]) / __img_std[2]

    return img

# endregion
'''
_13_prepare_annotation_frames_per_video_dict_untrimmed_multi_label_for_i3d: 从视频中进行帧采样
extract_features_i3d_charades: 通过I3D进行特征提取

Charades_v1_test.csv文件内容出错，替换如下位置的名字即可：
[x for x in data_test_csv if x not in data_test_json]
Out[58]: ['1.50E+08', '5.00E+07', '607']
[x for x in data_test_json if x not in data_test_csv]
Out[59]: ['00607', '150E6', '50E06']


'''
if __name__ == '__main__':
   
    # root_Charades_path = '/home/r/renpengzhen/Datasets/Charades' #1070服务器
    root_Charades_path = '/data/renpengzhen/data'  # 24G服务器
    n_frames_in = 512
    annotation_path = _13_prepare_annotation_frames_per_video_dict_untrimmed_multi_label_for_i3d(n_frames_in,root_Charades_path)
    extract_features_i3d_charades(root_Charades_path,n_frames_in, annotation_path)
    