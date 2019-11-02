#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014

# ./tools/generate_tsv.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out /coc/scratch/panderson43/tsv/nocaps/nocaps_val_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split nocaps_val

# ./tools/generate_tsv.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out /coc/scratch/panderson43/tsv/nocaps_36/nocaps_val_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split nocaps_val

#./tools/generate_tsv.py --gpu 0,1 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out /coc/scratch/panderson43/tsv/nocaps_36/nocaps_test_resnet101_faster_rcnn_genome_36.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split nocaps_test

#./tools/generate_tsv.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out /coc/scratch/panderson43/tsv/test2014/test2014_resnet101_faster_rcnn_genome.tsv.3 --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014

#./tools/generate_tsv.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out /coc/scratch/panderson43/tsv/nocaps/nocaps_test_resnet101_faster_rcnn_genome.tsv.4 --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split nocaps_test

#./tools/generate_tsv.py --gpu 0,1,2,3 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out /coc/scratch/panderson43/tsv/openimages/openimages_trainsubset_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split openimages_missing

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import jsonlines
import random
import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json
import pdb
import pandas as pd
import zlib

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 250

def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption","url"], usecols=range(0,2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df
def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

def get_box(im_h,im_w,region_num):
    box=[]
    dis=int(im_h/region_num)
    for y in range(0,im_h-2*dis+1,dis):
        box.append([0,y,im_w,y+dis])
    box.append([0,(region_num-1)*dis,im_w,im_h])
    return box
def load_image_ids(split_name, group_id, total_group, image_ids=None):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    bbox = []
    num_bbox = 6
    json_path='/home/lpx/bh/CUHK-PEDES/CUHK-PEDES'
    json_address=os.path.join(json_path,'reid_raw.json')
    count=0
    pre_id=None
    with open(json_address) as f:
        with jsonlines.open('train_cut8.jsoline','a') as train:
            with jsonlines.open('val_cut_8.jsonline','a') as val:
                with jsonlines.open('test_cut_8.jsonline','a') as test:
                    data=json.load(f)
                    for item in data:
                        file_path=os.path.join(json_path,'imgs/'+item['file_path'])
                        this_id=item['id']
                        splitt=item['split']
                        if (this_id != pre_id ) or pre_id == None:
                            count = 0
                            ids = int(this_id)*10 + count
                        else:
                            count += 1
                            ids = int(this_id)*10 + count
                        file_path=str(file_path)
                        img = cv2.imread(file_path)
                        im_h = img.shape[0]
                        im_w = img.shape[1]
                        box=get_box(im_h,im_w,num_bbox)
                        pre_id=this_id
                        split.append((file_path,ids))
                        bbox.append(box)
                        sentences = item['captions']
                        temp = {}
                        temp['sentences'] = sentences
                        temp['id'] = ids
                        temp['img_path'] = file_path
                        if splitt == 'train':
                            train.write(temp)
                        elif splitt == 'val':
                            val.write(temp)
                        elif splitt == 'test':
                            test.write(temp)
    return split, np.array(bbox), num_bbox

def get_detections_from_im(net, im, image_id, bbox=None, num_bbox=None, conf_thresh=0.2):

    if bbox is not None:
      scores, boxes, attr_scores, rel_scores = im_detect(net, im, bbox, force_boxes=True)
    else:
      scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data
    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(rois),
        'boxes': base64.b64encode(cls_boxes),
        'features': base64.b64encode(pool5), 
        'cls_prob': base64.b64encode(cls_prob)
    }

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='../models/vg/ResNet-101/faster_rcnn_end2end_final/test_gt.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default='../data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel', type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default='VCR_gt_resnet101_faster_rcnn_genome_cut8.tsv', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='../experiments/cfgs/faster_rcnn_end2end_resnet.yml', type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--total_group',
        help="the number of group for exracting",
        type=int,
        default=1
    )
    parser.add_argument(
        '--group_id',
        help=" group id for current analysis, used to shard",
        type=int,
        default=0
    )
    args = parser.parse_args()
    return args

def generate_tsv(gpu_id, prototxt, weights, image_ids, bbox, num_bbox, outfile):
    # First check if file exists, and if it is complete
    wanted_ids = set([image_id[1] for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                found_ids.add(item['image_id'])
    
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print 'GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids))
    else:
        print 'GPU {:d}: missing {:d}/{:d}'.format(gpu_id, len(missing), len(image_ids))
        print missing
    
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
            _t = {'misc' : Timer()}
            count = 0
            for ii, image in enumerate(image_ids):
                im_file,image_id = image
                if image_id in missing:
                    im = cv2.imread(im_file)
                    # if im is not None and min(im.shape[:2])>=200 and im.shape[2]==3:
                    _t['misc'].tic()
                    if bbox is not None:
                        writer.writerow(get_detections_from_im(net, im, image_id, np.array(bbox[ii]), num_bbox))
                    else:
                        writer.writerow(get_detections_from_im(net, im, image_id))
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print 'GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                              .format(gpu_id, count+1, len(missing), _t['misc'].average_time, 
                              _t['misc'].average_time*(len(missing)-count)/3600)
                    count += 1
                    # else:
                        # print 'image missing {:d}'.format(image_id)

if __name__ == '__main__':

    # merge_tsvs()
    # print fail
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids, bbox, num_bbox = load_image_ids(args.data_split, args.group_id, args.total_group)

    outfile = '%s.%d' % (args.outfile, args.group_id)
    generate_tsv(0, args.prototxt, args.caffemodel, image_ids, bbox, num_bbox, outfile)
