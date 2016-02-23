# -*- coding:utf-8 -*-  
import mxnet as mx
import mxnet as mx
import numpy as np
from collections import namedtuple
from skimage import io
from sklearn import utils
from random import randint
from random import shuffle
import matplotlib.pyplot as plt

def get_network(network_type):
    ''' 
     训练时使用not fully 的cnn。
     因为 fully cnn 没法高效传label 和gradient
     
     1)kitty dataset 有缺失和无效的disparity （整张图有差不多一半没有disparity）
     2)计算gradient 的时候不能直接用ndarray 而是numpy
     3)not fully cnn 训练时间和论文中的相符

     但是not fully cnn 之后需要扩展rnn as crf 就很麻烦了
    
     预测时可以使用fully cnn
    '''
    relu = {}
    conv = {}
    weight = {}
    bias = {}
    relu[0]  = mx.sym.Variable('left')
    relu[1]  = mx.sym.Variable('right')
    relu[2]  = mx.sym.Variable('left_downsample')
    relu[3]  = mx.sym.Variable('right_downsample')
    for num_layer in range(1,5):
        weight[0]   = mx.sym.Variable('l%d_blue' % num_layer)
        weight[1]   =  mx.sym.Variable('l%d_red' % num_layer)
        bias[0]    = mx.sym.Variable('bias%d_blue' % num_layer)
        bias[1]   = mx.sym.Variable('bias%d_red' % num_layer)
        if num_layer<=2:
            kernel = (3,3)
            pad = (0,0)
            num_filter = 32
        else:
            kernel = (5,5)
            pad = (0,0)
            num_filter = 200
        for j in range(4):
            conv[j]  = mx.sym.Convolution(data = relu[j] ,weight=weight[j/2],bias=bias[j/2],kernel=kernel,num_filter=num_filter,pad= pad)
            relu[j] =  mx.sym.Activation(data=conv[j], act_type="relu")
    
    if network_type!='fully':
        flatten = {}        
        for j in range(4):
            flatten[j] = mx.sym.Flatten(data=relu[j])
        s = mx.sym.Dotproduct(data1=flatten[0],data2=flatten[1])
        net = mx.sym.Group([flatten[0],flatten[1],s])
    else:
        net  = mx.sym.Group([relu[0],relu[1]])
    return net

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])
class img_iter(mx.io.DataIter):

    def __init__(self,img_dir,ctx):
        self.img_dir = img_dir
        self.num_imgs = len(img_dir)
        self.ctx = ctx
        self.reset()

    def reset(self):
        self.img_idx = -1

    def iter_next(self):
        self.img_idx += 1
        if self.img_idx >=  self.num_imgs:
            return False
        else:
            return True
    
    def getdata(self):

        left = io.imread(self.img_dir[self.img_idx][1]) - 128.0
        left = mx.nd.array(left.swapaxes(2,1).swapaxes(1,0),ctx)
        
        right= io.imread(self.img_dir[self.img_idx][2]) - 128.0
        right= mx.nd.array(right.swapaxes(2,1).swapaxes(1,0),ctx)
        
        return [left,right]

    def getlabel(self):
        dis  = mx.nd.array(np.round(io.imread(self.img_dir[self.img_idx][0])/256.0).astype(int),self.ctx)
        return dis

    def getindex(self):
        return self.index
    
    def getpad(self):
        return 0



'''       
class Dot(mx.operator.NumpyOp):
    def __init__(self):
        super(Dot,self).__init__(True)
    def list_arguments(self):
        return ['x','y']
    def list_outputs(self):
        return ['output']
    def infer_shape(self,in_shape):
        data1_shape = in_shape[0]
        data2_shape = in_shape[1]
        output_shape = (in_shape[0][0],) 
        return [data1_shape,data2_shape],[output_shape]
    def forward(self,in_data,out_data):
        x = in_data[0]
        y = in_data[1]
        output = out_data[0]
        output[:] =  (x*y).sum(axis=1) 
    def backward(self,out_grad,in_data,out_data,in_grad):
        x = in_data[0]
        y = in_data[1]
        dx = in_grad[0]
        dy = in_grad[1]
        for i in xrange(out_grad[0].shape[0]):
            dx[i] = out_grad[0][i] * y[i]
            dy[i] = out_grad[0][i] * x[i]
'''