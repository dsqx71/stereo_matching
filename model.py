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
class dataiter(mx.io.DataIter):

    def __init__(self,img_dir,batch_size,ctx,low,high,datatype='train'):
        '''
            img_dir: 图片位置 [(disparity_dir,left_dir,right_dir),.......]
            low  high 决定负样本的相对于正样本的位置
        '''
        self.batch_size = batch_size
        self.reset()
        self.img_dir  = img_dir
        self.num_imgs = len(img_dir)
        self.datatype = datatype   
        self.ctx = ctx
        self.low = low
        self.high = high
     
    def produce_patch(self,ith):
        '''
            self.l_ls : list of left patch
        '''
        dis  = np.round(io.imread(self.img_dir[ith][0])/256.0).astype(int)
        left = io.imread(self.img_dir[ith][1]) - 128.0
        left =  left.swapaxes(2,1).swapaxes(1,0)
        right= io.imread(self.img_dir[ith][2]) - 128.0
        right= right.swapaxes(2,1).swapaxes(1,0)
        self.generate_patch_with_ground_truth(left,right,dis)  
        utils.shuffle(self.l_ls,self.r_ls,self.ld_ls,self.rd_ls,self.labels)
        
    def generate_patch_with_ground_truth(self,left,right,dis):
        '''
            generate patch from a img
        '''
        scale = 6
        for y in xrange(scale,dis.shape[0]-scale):
            for x in xrange(scale,dis.shape[1]-scale):
                if dis[y,x]!=0:
                    d = dis[y,x]
                    if x-d>=scale :
                        self.l_ls.append(left[:,y-scale:y+1+scale,x-scale:x+1+scale])
                        self.r_ls.append(right[:,y-scale:y+1+scale,x-scale-d:x+1+scale-d])
                        self.ld_ls.append(left[:,y-scale:y+1+scale:2,x-scale:x+1+scale:2])
                        self.rd_ls.append(right[:,y-scale:y+1+scale:2,x-scale-d:x+1+scale-d:2])
                        self.labels.append(1)
                        while True:
                            temp = [x - d + move for move in range(self.low,self.high+1)]
                            temp.extend([x - d - move for move in range(self.low,self.high+1)])
                            xn = np.random.choice(temp)
                            if xn<dis.shape[1]-scale and x-d != xn and xn>=scale:
                                break
                        self.l_ls.append( left[:,y-scale:y+1+scale,    x-scale:x+1+scale])
                        self.r_ls.append(right[:,y-scale:y+1+scale,xn-scale:xn+1+scale])
                        self.ld_ls.append(left[:,y-scale:y+scale+1:2,x-scale:x+1+scale:2])
                        self.rd_ls.append(right[:,y-scale:y+scale+1:2,xn-scale:xn+1+scale:2])
                        self.labels.append(0) 
                        self.inventory +=2      
    
    def reset(self):
        '''
          这几个list保存patch，inventory 表示dataiter剩余的patch数量
        '''
        self.index = 0
        self.img_idx = 0
        self.inventory = 0
        self.l_ls = []
        self.r_ls = []
        self.ld_ls = []
        self.rd_ls = []
        self.labels = []

    def iter_next(self):
        if self.inventory < self.batch_size:
            if self.img_idx >= self.num_imgs:
                return False
            if self.datatype !='test':
                self.produce_patch(self.img_idx)
            else:
                self.produce_patch_test(self.img_idx) 
                #没写
            self.img_idx+=1
            return self.iter_next()
        else: 
            self.inventory -= self.batch_size
            return True

    def getdata(self):

        left = mx.nd.array(np.asarray(self.l_ls[:self.batch_size]),self.ctx)
        right = mx.nd.array(np.asarray(self.r_ls[:self.batch_size]),self.ctx)
        left_downsample = mx.nd.array(np.asarray(self.ld_ls[:self.batch_size]),self.ctx)
        right_downsample = mx.nd.array(np.asarray(self.rd_ls[:self.batch_size]),self.ctx)
        del self.l_ls[:self.batch_size]
        del self.r_ls[:self.batch_size]
        del self.ld_ls[:self.batch_size]
        del self.rd_ls[:self.batch_size]
        
        return [left,right,left_downsample,right_downsample]
    
    def getlabel(self):
        if self.datatype !='test':
            result =  mx.nd.array(self.labels[:self.batch_size])
            del self.labels[:self.batch_size]
            return result
        else :
            return None
    
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