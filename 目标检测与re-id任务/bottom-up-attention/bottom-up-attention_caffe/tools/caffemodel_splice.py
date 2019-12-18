#coding=utf-8
import _init_paths
import caffe
import collections
from collections import OrderedDict
# caffe.set_mode_cpu
import copy
def caffe_net(prototxt,caffemodel):
    return caffe.Net(prototxt, caffemodel, caffe.TEST)
def weight_to_prototxt_net(keys,net1,net2):
    '''
    将net1中关键字为keys的权重片段复制到net2中
    :param keys: 要进行复制权重层的关键字片段
    :param net1: 被复制的网络
    :param net2: 目标网络
    :return:
    '''
    for key in keys:
        n_params = len(net1.params[key])
        try:
            for i in range(n_params):
                net2.params[key][i].data[...] = net1.params[key][i].data[...]
        except Exception as e:
            print(e)
    return net2
    

if __name__ == '__main__':
    data_name = 'CUHK03'
    # data_name = 'market-1501'
    split = ['scale4f_branch2c',"rpn_conv/3x3","rpn_bbox_pred","res5a_branch1",'scale5c_branch2c',"cls_score"]
    '''
    层名片段                                 :  参数属于的模型
    [                : 'scale4f_branch2c']  :  resnet50
    ['rpn_conv/3x3'  : 'rpn_bbox_pred'   ]  :  resnet101
    ['res5a_branch1' : 'scale5c_branch2c']  :  resnet50
    ['cls_score'     :                   ]  :  resnet101
    '''
    #改修好的框架：变backbone为resnet50的模型结构
    faster_rcnn_resnet50_prototxt = '/home/lpx/renpengzhen/bottom-up-attention1/models/vg/ResNet-101/faster_rcnn_end2end_final/test_resNet50_RCNN.prototxt' #替换掉resnet101为resnet50的配置文件,最终需要的模型
    faster_rcnn_50_keys = caffe.Net(faster_rcnn_resnet50_prototxt, caffe.TEST).params.keys()

    
    resnet50_weight,output_model_path = '',''
    if data_name == 'CUHK03':
        resnet50_weight = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ResNet-50-model.caffemodel' #CUHK03原始的权重
        output_model_path = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ouput_model/test_resNet50_CUHK03.caffemodel' #输出权重
    elif data_name == 'market-1501':
        resnet50_weight = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ResNet_50_IDE_iter_50000.caffemodel' #在market-1501数据集上的原始的权重
        output_model_path = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ouput_model/test_resNet50_Market-1501.caffemodel' #输出权重
    # resnet50
    resnet50_prototxt = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ResNet_50_test.prototxt'
    resnet50_net = caffe_net(resnet50_prototxt, resnet50_weight)
    resnet50_keys = resnet50_net.params.keys()
    
    #resnet101
    resnet101_prototxt = '/home/lpx/renpengzhen/bottom-up-attention1/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
    resnet101_weight = '/home/lpx/renpengzhen/bottom-up-attention1/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
    resnet101_net = caffe_net(resnet101_prototxt, resnet101_weight)
    resnet101_keys = resnet101_net.params.keys()

    #split = ['scale4f_branch2c',"rpn_conv/3x3","rpn_bbox_pred","res5a_branch1",'scale5c_branch2c',"cls_score"]
    
    #第1部分权重:[                : 'scale4f_branch2c']
    split1_idx = resnet50_keys.index(split[0])+1
    faster_rcnn_resnet50_1 = resnet50_keys[:split1_idx] #修改过之后模型的第一部分参数

    pro_1 = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ResNet_50_test_frist.prototxt'
    net1 = caffe_net(pro_1, resnet50_weight)
    net1.save(
        '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ouput_model/ResNet_50_test_frist.caffemodel')
    net1_weight = '/home/lpx/renpengzhen/bottom-up-attention1/resnet50/ouput_model/ResNet_50_test_frist.caffemodel'
    faster_rcnn_resnet50_1_net = caffe_net(pro_1, net1_weight)
    faster_rcnn_resnet50_1_net_keys = faster_rcnn_resnet50_1_net.params.keys()
    assert faster_rcnn_resnet50_1_net_keys == faster_rcnn_resnet50_1

    #第2部分权重:['rpn_conv/3x3'  : 'rpn_bbox_pred'] : resnet101
    split2_0,split2_1 = split[1],split[2]
    split2_0_index,split2_1_index = resnet101_keys.index(split2_0),resnet101_keys.index(split2_1)
    faster_rcnn_resnet50_2 = resnet101_keys[split2_0_index:split2_1_index+1]
    faster_rcnn_resnet50_empty = caffe.Net(faster_rcnn_resnet50_prototxt, net1_weight, caffe.TEST) #后面3部分的权重是空的
    net2 = weight_to_prototxt_net(faster_rcnn_resnet50_2,resnet101_net,faster_rcnn_resnet50_empty)
    
    #第3部分权重:['res5a_branch1' : 'scale5c_branch2c']  :  resnet50
    split3_0, split3_1 = split[3], split[4]
    split3_0_index, split3_1_index = resnet50_keys.index(split3_0), resnet50_keys.index(split3_1)
    faster_rcnn_resnet50_3 = resnet50_keys[split3_0_index:split3_1_index + 1]
    net3 = weight_to_prototxt_net(faster_rcnn_resnet50_3, resnet50_net, net2)

    # 第4部分权重:['cls_score'    :                   ]   :  resnet101
    split4_index = resnet101_keys.index(split[5])
    faster_rcnn_resnet50_4 = resnet101_keys[split4_index:]
    net4 = weight_to_prototxt_net(faster_rcnn_resnet50_4, resnet101_net, net3)

    faster_rcnn_resnet50_keys = faster_rcnn_resnet50_1+faster_rcnn_resnet50_2+faster_rcnn_resnet50_3+faster_rcnn_resnet50_4
    #判断prototext的拼接是否正确
    assert faster_rcnn_resnet50_keys == faster_rcnn_50_keys
    

    #模型参数保存
    
    #
    net4.save(output_model_path)
    
    #测试
    faster_rcnn_resnet50 = caffe.Net(faster_rcnn_resnet50_prototxt, output_model_path, caffe.TEST)
    faster_rcnn_resnet50_keys = faster_rcnn_resnet50.params.keys()
    for key in faster_rcnn_resnet50_1+faster_rcnn_resnet50_3:
        # print('=====================')
        # print(faster_rcnn_resnet50.params[key][0].data)
        # print('---------------------')
        # print(resnet50_net.params[key][0].data)
        assert faster_rcnn_resnet50.params[key][0].data.all()==resnet50_net.params[key][0].data.all()
    for key in faster_rcnn_resnet50_2+faster_rcnn_resnet50_4:
        # print('=====================')
        # print(faster_rcnn_resnet50.params[key][0].data)
        # print('---------------------')
        # print(resnet101_net.params[key][0].data)
        assert faster_rcnn_resnet50.params[key][0].data.all()==resnet101_net.params[key][0].data.all()


    print(len(faster_rcnn_resnet50_keys))

