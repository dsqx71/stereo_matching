import mxnet as mx
import numpy as np
import logging
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
        print  out_data.shape 
    def backward(self,out_grad,in_data,out_data,in_grad):
        x = in_data[0]
        y = in_data[1]
        dx = in_grad[0]
        dy = in_grad[1]
        dx = out_grad.mean() * y
        dy = out_grad.mean() * x
        print  x.shape == dx.shape

left  = mx.sym.Variable('left')
right = mx.sym.Variable('right')
leftdownsample = mx.sym.Variable('left_downsample')
rightdownsample= mx.sym.Variable('right_downsample')
label = mx.sym.Variable('label')
weight1_blue = mx.sym.Variable('l1_blue')
weight2_blue = mx.sym.Variable('l2_blue')
weight3_blue = mx.sym.Variable('l3_blue')
weight4_blue = mx.sym.Variable('l4_blue')
weight1_red  = mx.sym.Variable('l1_red')
weight2_red  = mx.sym.Variable('l2_red')
weight3_red  = mx.sym.Variable('l3_red')
weight4_red  = mx.sym.Variable('l4_red')
conv1_1_blue = mx.sym.Convolution(data=left, weight=weight1_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
conv1_2_blue = mx.sym.Convolution(data=right,weight=weight1_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
conv2_1_blue = mx.sym.Convolution(data=conv1_1_blue,weight=weight2_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
conv2_2_blue = mx.sym.Convolution(data=conv1_2_blue,weight=weight2_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
conv3_1_blue = mx.sym.Convolution(data=conv2_1_blue,weight=weight3_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
conv3_2_blue = mx.sym.Convolution(data=conv2_2_blue,weight=weight3_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
conv4_1_blue = mx.sym.Convolution(data=conv3_1_blue,weight=weight4_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
conv4_2_blue = mx.sym.Convolution(data=conv3_2_blue,weight=weight4_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
conv1_1_red = mx.sym.Convolution(data=leftdownsample,weight=weight1_red,kernel=(3,3),pad=(1,1),num_filter = 32)
conv1_2_red = mx.sym.Convolution(data=rightdownsample,weight=weight1_red,kernel=(3,3),pad=(1,1),num_filter =32)
conv2_1_red = mx.sym.Convolution(data=conv1_1_red,weight=weight2_red,kernel=(3,3),pad=(1,1),num_filter = 32)
conv2_2_red = mx.sym.Convolution(data=conv1_2_red,weight=weight2_red,kernel=(3,3),pad=(1,1),num_filter = 32)
conv3_1_red = mx.sym.Convolution(data=conv2_1_red,weight=weight3_red,kernel=(5,5),pad=(2,2),num_filter = 200)
conv3_2_red = mx.sym.Convolution(data=conv2_2_red,weight=weight3_red,kernel=(5,5),pad=(2,2),num_filter = 200)
conv4_1_red = mx.sym.Convolution(data=conv3_1_red,weight=weight4_red,kernel=(5,5),pad=(2,2),num_filter = 200)
conv4_2_red = mx.sym.Convolution(data=conv3_2_red,weight=weight4_red,kernel=(5,5),pad=(2,2),num_filter = 200)
f_blue1 = mx.sym.Flatten(data = conv4_1_blue)
f_blue2 = mx.sym.Flatten(data = conv4_2_blue)
f_red1 = mx.sym.Flatten(data = conv4_1_red)
f_red2 = mx.sym.Flatten(data = conv4_2_red)
dot = Dot()
#w1 = mx.sym.Variable('w1')
#w2 = mx.sym.Variable('w2')
s = dot(x = f_blue1 ,y =f_blue2,name='dot_product1') 
output = mx.sym.LinearRegressionOutput(data = s, label = label, name='LinearRegression' )


def init(key,weight):
    if 'bias' in key:
        weight[:] = 0
    elif key not in data_sign:
        weight[:] = mx.random.uniform(-.05,.05,weight.shape)  

dis_iter = mx.io.ImageRecordIter(path_imgrec = 'dis.rec',data_shape=(3,369,1225), batch_size= 3)
left_iter = mx.io.ImageRecordIter(path_imgrec = 'left.rec',data_shape=(3,369,1225), batch_size= 3)
right_iter = mx.io.ImageRecordIter(path_imgrec = 'right.rec',data_shape=(3,369,1225), batch_size= 3)
data_iter = mx.io.PrefetchingIter([dis_iter,left_iter,right_iter])
net = output

data_sign = ['left','right','left_downsample','right_downsample','label']
batch_size = 100
s1 = (batch_size,3,13,13)
s2 = (batch_size,3,7,7)
num_epoches = 1
args_shape,out_shape,aux_shape = net.infer_shape(left=s1,right=s1,left_downsample=s2,right_downsample=s2)
args_shape = dict(zip(net.list_arguments(),args_shape))
print args_shape
executor = net.simple_bind(ctx=mx.gpu(3),grad_req='write',left = s1,right= s1,left_downsample=s2,right_downsample=s2)
args = dict(zip(net.list_arguments(), executor.arg_arrays))
grads = dict(zip(net.list_arguments(), executor.grad_arrays))
outputs = dict(zip(net.list_outputs(), executor.outputs))

for key in args:
    init(key,args[key])

l_ls =[]
r_ls =[]
ld_ls=[]
rd_ls=[]

for j in xrange(nums_epoch):  
    data_iter.reset()
    for batch in data_iter:
        dis = batch.data[0].asnumpy()[0]
        left= batch.data[1].asnumpy()[0]
        right=batch.data[2].asnumpy()[0]
        for y in xrange(dis.shape[1]):
            for x in xrange(dis.shape[2]):
                if dis[0,y,x]!=0 and x-dis[0,y,x]>=6 and x<dis.shape[2]-6 and y>=6 and y<dis.shape[1]-6:
                    d = dis[0,y,x]
                    l_ls.append(left[:,y-6:y+7,x-6:x+7])
                    r_ls.append(right[:,y-6:y+7,x-6-d:x+7-d])
                    ld_ls.append(left[:,y-6:y+7:2,x-6:x+7:2])
                    rd_ls.append(right[:,y-6:y+7:2,x-6-d:x+7-d:2])
                    labels.append(1)
                    while True:
                        xn = np.random.randint(6,dis.shape[2]-7)
                        if x-d != xn:
                            break
                    l_ls.append(left[:,y-6:y+7,x-6:x+7])
                    r_ls.append(right[:,y-6:y+7,xn-6:xn+7])
                    ld_ls.append(left[:,y-6:y+7:2,x-6:x+7:2])
                    rd_ls.append(right[:,y-6:y+7:2,xn-6:xn+7:2])
                    labels.append(0)
        while len(labels)>=batch_size:
            args['label'][:] = np.asarray(labels[:batch_size])
            del labels[:batch_size]
            args['left'][:] = np.asarray(l_ls[:batch_size])
            del l_ls[:batch_size]
            args['right'][:]=np.asarray(r_ls[:batch_size])
            del r_ls[:batch_size]
            args['left_downsample'] = np.asarray(ld_ls[:batch_size])
            del ld_ls[:batch_size]
            args['right_downsample']= np.asarray(rd_ls[:batch_size])
            del rd_ls[:batch_size]
            executor.forward(is_train=True)
