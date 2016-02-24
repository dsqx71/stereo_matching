#-*- coding:utf-8 -*-
from skimage import io
import mxnet as mx
from random import shuffle,randint
from model import get_network
import matplotlib.pyplot as plt
import numpy as np

def output_embedding(left_dir,right_dir,epoch):  
    left = io.imread(left_dir).swapaxes(2,1).swapaxes(1,0) - 128.0
    right= io.imread(right_dir).swapaxes(2,1).swapaxes(1,0) - 128.0
    s = (1,3,left.shape[1],left.shape[2])
    ctx = mx.gpu(3)
    net,executor =  load_model('stereo',epoch,s,'fully',ctx)
    args  = dict(zip(net.list_arguments(),executor.arg_arrays))
    args['left'][:] = np.array([left])
    args['right'][:] = np.array([right])
    executor.forward(is_train=False)
    return io.imread(left_dir),io.imread(right_dir),executor.outputs[0].asnumpy()[0],executor.outputs[1].asnumpy()[0]

def get_kitty_data_dir(low,high):
    img_dir = []
    for num in range(low,high):
        dir_name = '000{}'.format(num)
        if len(dir_name) ==4 :
            dir_name = '00'+dir_name
        elif len(dir_name) == 5:
            dir_name = '0'+dir_name
        gt = './disp_noc/'+dir_name+'_10.png'.format(num)
        imgL = './colored_0/'+dir_name+'_10.png'.format(num)
        imgR = './colored_1/'+dir_name+'_10.png'.format(num)
        img_dir.append((gt,imgL,imgR))
    #shuffle(img_dir)
    return img_dir

def load_model(name,epoch,shape,network_type,ctx):

    data_sign = ['left','right','left_downsample','right_downsample','label','LinearRegression_label','gt']
    net,args,aux = mx.model.load_checkpoint(name,epoch)
    keys = net.list_arguments()
    net = get_network(network_type)
    executor = net.simple_bind(ctx=ctx,grad_req='add',left = shape,right= shape)
    for key in executor.arg_dict:
        if key in  data_sign:
            executor.arg_dict[key][:] = mx.nd.zeros((executor.arg_dict[key].shape),ctx)
        else:
            if key in args:
                executor.arg_dict[key][:] = args[key]
            else:
                init(key,executor.arg_dict[key])
    return net,executor

def draw_patch(args,executor,img_idx):
    '''
        画patch，附带groud truth 和 matching score
    '''
    idx = randint(30,1000)
    l_p =  args['left'].asnumpy()[idx].swapaxes(0,1).swapaxes(1,2) + 128
    r_p = args['right'].asnumpy()[idx].swapaxes(0,1).swapaxes(1,2) + 128
    result = (executor.outputs[2].asnumpy()[idx],args['gt'].asnumpy()[idx])
    
    plt.figure()
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)
    p1.imshow(l_p)
    p2.imshow(r_p)
    plt.title('gt: %d  matching score: %.5f ' % (result[1],result[0]))
    plt.savefig('./result/img_%d_gt_%d_matchingscore_%.5f.jpg' % (img_idx,result[1],result[0]))
    plt.close()   
