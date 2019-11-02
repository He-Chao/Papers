#coding=utf-8
import _init_paths
import caffe
import collections
from collections import OrderedDict
caffe.set_mode_cpu
import copy

if __name__ == '__main__':
    resnet50_net_final = '/home/lpx/renpengzhen/bottom-up-attention1/models/vg/ResNet-101/faster_rcnn_end2end_final/test_resNet50.prototxt' #替换掉resnet101为resnet50的配置文件,最终需要的模型
    resnet50_net = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ResNet_50_test.prototxt'#原始的模型结构
    resnet50_weight = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ResNet-50-model.caffemodel' #原始的权重
    resnet50_net1 = caffe.Net(resnet50_net, resnet50_weight, caffe.TEST)
    resnet50_keys = resnet50_net1.params.keys()
    split_101_50_par = 'scale4f_branch2c'
    split_101_50_index = resnet50_keys.index(split_101_50_par)+1
    resnet50_net_frist = resnet50_keys[:split_101_50_index] #修改过之后模型的第一部分参数
    
    #原始的resnet101的模型及对应的权重
    resnet101_net = '/home/lpx/renpengzhen/bottom-up-attention1/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
    resnet101_weight = '/home/lpx/renpengzhen/bottom-up-attention1/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
    resnet101_net1 = caffe.Net(resnet101_net, resnet101_weight, caffe.TEST)
    resnet101_keys = resnet101_net1.params.keys()
    split_101_rpn_par = 'rpn_conv/3x3'
    split_101_rpn_index = resnet101_keys.index(split_101_rpn_par)
    resnet101_net_end = resnet101_keys[split_101_rpn_index:]  # 修改过之模型最后三部分的参数
    faster_rcnn_final = resnet50_net_frist+resnet101_net_end
    
    # print(faster_rcnn_final)
    
    #只保留resnet50前半部分的权重和对应的model
    ResNet_50_test_frist = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ResNet_50_test_frist.prototxt'
    ResNet_50_test_frist_net = caffe.Net(ResNet_50_test_frist, resnet50_weight, caffe.TEST)
    # print(type(ResNet_50_test_frist_net)) #<class 'caffe._caffe.Net'>
    ResNet_50_test_frist_net.save('/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ouput_model/ResNet_50_test_frist.caffemodel')
    ResNet_50_test_frist_weight = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ouput_model/ResNet_50_test_frist.caffemodel'
    ResNet_50_test_frist_net1 = caffe.Net(ResNet_50_test_frist, ResNet_50_test_frist_weight, caffe.TEST)
    ResNet_50_test_frist_net1_keys = ResNet_50_test_frist_net1.params.keys()
    # print (type(ResNet_50_test_frist_net1.params)) #collections.OrderedDict
    # print(faster_rcnn_final)
    # print(type(ResNet_50_test_frist_net1.params))
    # print(ResNet_50_test_frist_net1_keys)

    #将resnet101后半部分的权重赋值到ResNet_50_test_final上
    ResNet_50_test_final = caffe.Net(resnet50_net_final,ResNet_50_test_frist_weight,caffe.TEST)
    for key in resnet101_net_end:
        n_params = len(resnet101_net1.params[key])
        try:
            for i in range(n_params):
                ResNet_50_test_final.params[key][i].data[...] = resnet101_net1.params[key][i].data[...]
        except Exception as e:
            print(e)
    

    #模型参数保存
    ResNet_50_test_final.save('/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ouput_model/test_resNet50.caffemodel')
    
    
    #测试
    test_resNet50_weight = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ouput_model/test_resNet50.caffemodel'
    resnet50_net_final1 = caffe.Net(resnet50_net_final, test_resNet50_weight, caffe.TEST)
    resnet50_net_final1_keys = resnet50_net_final1.params.keys()
    for key in resnet50_net_frist:
        print('=================================')
        print(key)
        print(resnet50_net1.params[key][0].data)
        print('---------------------------------')
        print(resnet50_net_final1.params[key][0].data)
    for key in resnet101_net_end:
        print('=================================')
        print(key)
        print(resnet101_net1.params[key][0].data)
        print('---------------------------------')
        print(resnet50_net_final1.params[key][0].data)

    print(len(resnet50_net_final1_keys))

