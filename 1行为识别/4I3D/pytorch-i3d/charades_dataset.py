import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
    #对小的视频帧进行放大
  #vid:视频的名字，start：开始的帧数，num=64
  
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]] #读取图片，并将图片的第三个维度调换顺序
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h) #扩大比例
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc) #使用cv2.resize放大图片
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=157):
    # make_dataset('charades/charades.json', 'training', root, mode)
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        #vid：视频的名字
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            #如果在原始的视频数据集中不存在，则跳过
            continue
        num_frames = len(os.listdir(os.path.join(root, vid))) #得到该视频的帧数
        if mode == 'flow':
            num_frames = num_frames//2
            
        if num_frames < 66:
            #帧数小于66的视频跳过
            continue

        label = np.zeros((num_classes,num_frames), np.float32) #为视频的每一帧的定义一个标签

        fps = num_frames/data[vid]['duration'] #计算帧频
        #给每一帧标定标签
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    # ann[1]为该动作开始的时间，ann[2]为该动作结束的时间
                    label[ann[0], fr] = 1 # binary classification ，ann[0]该帧所属的类别
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1
    
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None):
        # dataset = Dataset(train_split, 'training', root, mode, train_transforms),
        # train_split='charades/charades.json'
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        #vid:视频的名字，label:每一帧的标签，dur:视频的长度，nf:视频的帧数
        start_f = random.randint(1,nf-65) #随机挑选视频帧开始的位置

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64) #对小于226的视频帧进行放大
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64)
        label = label[:, start_f:start_f+64]

        imgs = self.transforms(imgs) #对视频帧进行中心裁剪

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
