
import mxnet as mx
import numpy as np
import logging
import math
from skimage import io
from sklearn import utils
logging.basicConfig(level = logging.DEBUG)


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

data_sign = ['left','right','left_downsample','right_downsample','label','LinearRegression_label']
left  = mx.sym.Variable('left')
right = mx.sym.Variable('right')

leftdownsample = mx.sym.Variable('left_downsample')
rightdownsample= mx.sym.Variable('right_downsample')
weight1_blue = mx.sym.Variable('l1_blue')
weight2_blue = mx.sym.Variable('l2_blue')
weight3_blue = mx.sym.Variable('l3_blue')
weight4_blue = mx.sym.Variable('l4_blue')
b1_blue = mx.sym.Variable('bias1_blue')
b2_blue = mx.sym.Variable('bias2_blue')
b3_blue = mx.sym.Variable('bias3_blue')
b4_blue = mx.sym.Variable('bias4_blue')
weight1_red  = mx.sym.Variable('l1_red')
weight2_red  = mx.sym.Variable('l2_red')
weight3_red  = mx.sym.Variable('l3_red')
weight4_red  = mx.sym.Variable('l4_red')
b1_red = mx.sym.Variable('bias1_red')
b2_red = mx.sym.Variable('bias2_red')
b3_red = mx.sym.Variable('bias3_red')
b4_red = mx.sym.Variable('bias4_red')

conv1_1_blue = mx.sym.Convolution(data=left, weight=weight1_blue,bias =b1_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
conv1_2_blue = mx.sym.Convolution(data=right,weight=weight1_blue,bias =b1_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
conv2_1_blue = mx.sym.Convolution(data=conv1_1_blue,weight=weight2_blue,bias = b2_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
conv2_2_blue = mx.sym.Convolution(data=conv1_2_blue,weight=weight2_blue,bias = b2_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
conv3_1_blue = mx.sym.Convolution(data=conv2_1_blue,weight=weight3_blue,bias = b3_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
conv3_2_blue = mx.sym.Convolution(data=conv2_2_blue,weight=weight3_blue,bias = b3_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
conv4_1_blue = mx.sym.Convolution(data=conv3_1_blue,weight=weight4_blue,bias = b4_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
conv4_2_blue = mx.sym.Convolution(data=conv3_2_blue,weight=weight4_blue,bias = b4_blue,kernel=(5,5),pad=(2,2),num_filter = 200)

conv1_1_red = mx.sym.Convolution(data=leftdownsample,weight=weight1_red,bias = b1_red,kernel=(3,3),pad=(1,1),num_filter = 32)
conv1_2_red = mx.sym.Convolution(data=rightdownsample,weight=weight1_red,bias = b1_red,kernel=(3,3),pad=(1,1),num_filter =32)
conv2_1_red = mx.sym.Convolution(data=conv1_1_red,weight=weight2_red,bias = b2_red,kernel=(3,3),pad=(1,1),num_filter = 32)
conv2_2_red = mx.sym.Convolution(data=conv1_2_red,weight=weight2_red,bias = b2_red,kernel=(3,3),pad=(1,1),num_filter = 32)
conv3_1_red = mx.sym.Convolution(data=conv2_1_red,weight=weight3_red,bias = b3_red,kernel=(5,5),pad=(2,2),num_filter = 200)
conv3_2_red = mx.sym.Convolution(data=conv2_2_red,weight=weight3_red,bias = b3_red,kernel=(5,5),pad=(2,2),num_filter = 200)
conv4_1_red = mx.sym.Convolution(data=conv3_1_red,weight=weight4_red,bias = b4_red,kernel=(5,5),pad=(2,2),num_filter = 200)
conv4_2_red = mx.sym.Convolution(data=conv3_2_red,weight=weight4_red,bias = b4_red,kernel=(5,5),pad=(2,2),num_filter = 200)
f_blue1 = mx.sym.Flatten(data = conv4_1_blue)
f_blue2 = mx.sym.Flatten(data = conv4_2_blue)
f_red1 = mx.sym.Flatten(data = conv4_1_red)
f_red2 = mx.sym.Flatten(data = conv4_2_red)
s = mx.sym.Dotproduct(data1=f_blue1,data2=f_blue2)
net = mx.sym.Group([f_blue1,f_blue2,s])

batch_size = 1024
s1 = (batch_size,3,13,13)
s2 = (batch_size,3,7,7)

args_shape,out_shape,aux_shape = net.infer_shape(left=s1,right=s1)
args_shape = dict(zip(net.list_arguments(),args_shape))
executor = net.simple_bind(ctx=mx.gpu(3),grad_req='write',left = s1,right= s1)
keys = net.list_arguments()
args = executor.arg_arrays
grads = dict(zip(net.list_arguments(),executor.grad_arrays))
args = dict(zip(keys,args))
auxs = dict(zip(keys,executor.arg_arrays))

mx.viz.plot_network(net)


def get_data_dir(high,low):
    data = []
    for num in range(high,low):
        dir_name = '000{}'.format(num)
        if len(dir_name) ==4 :
            dir_name = '00'+dir_name
        elif len(dir_name) == 5:
            dir_name = '0'+dir_name
        gt = './disp_noc/'+dir_name+'_10.png'.format(num)
        imgL = './colored_0/'+dir_name+'_10.png'.format(num)
        imgR ='./colored_0/'+dir_name+'_11.png'.format(num)
        data.append((gt,imgL,imgR))
    return data

def init(key,weight):
    if 'bias' in key:
        weight[:] = 0
    elif key not in data_sign:
        weight[:] = mx.random.uniform(-0.007,0.007,weight.shape) 
        
def sgd(key,weight,grad,lr,bacth_size):
    if key not in data_sign:
        weight = weight - lr*(1/batch_size)*grad

def load_data():
    args['left'][:] = np.asarray(l_ls[:batch_size])
    del l_ls[:batch_size]
    args['right'][:]= np.asarray(r_ls[:batch_size])
    del r_ls[:batch_size]
    args['left_downsample'] = mx.nd.array(ld_ls[:batch_size])
    del ld_ls[:batch_size]
    args['right_downsample']= mx.nd.array(rd_ls[:batch_size])
    del rd_ls[:batch_size]
    gt = mx.nd.array(labels[:batch_size])
    del labels[:batch_size]
    
def cal_grads(pred,label):
    return mx.nd.array((pred - label)*2)

def train():
    utils.shuffle(l_ls,r_ls,ld_ls,rd_ls,labels,random_state=0)
    global count
    global tot
    while len(labels)>=batch_size:  
        load_data() 
        count +=1
        executor.forward(is_train=True)
        output = executor.outputs[2]
        grad = 2*(output-gt)
        grad = grad.copyto(mx.gpu(3))
        loss = (mx.nd.square(output-gt).asnumpy()).mean()
        logging.info("{}th pair img:{}th iteration square loss:{}".format(num,count,loss))
        executor.backward([mx.nd.zeros((batch_size,33800),ctx=mx.gpu(3)),mx.nd.zeros((batch_size,33800),ctx=mx.gpu(3)),grad])
        for index,key in enumerate(keys):
            if key not in data_sign:
                opt.update(index,args[key],grads[key],states[key])
        tot+=1
def valdiate():
    global count
    global tot
    utils.shuffle(l_ls,r_ls,ld_ls,rd_ls,labels,random_state=0)
    while len(labels)>=batch_size:  
        load_data() 
        count +=1
        executor.forward(is_train=False)
        output = executor.outputs[2]
        loss = (mx.nd.square(output-gt).asnumpy()).mean()
        logging.info("validate: {}th pair img:{}th iteration square loss:{}".format(num,count,loss))
        tot+=1
def generate_patch(left,right,dis):
    for y in xrange(scale,dis.shape[0]-scale):
        for x in xrange(scale,dis.shape[1]-scale):
            if dis[y,x]!=0:
                d = dis[y,x]
                if x-d>=scale :
                    l_ls.append(left[:,y-scale:y+1+scale,x-scale:x+1+scale])
                    r_ls.append(right[:,y-scale:y+1+scale,x-scale-d:x+1+scale-d])
                    ld_ls.append(left[:,y-scale:y+1+scale:2,x-scale:x+1+scale:2])
                    rd_ls.append(right[:,y-scale:y+1+scale:2,x-scale-d:x+1+scale-d:2])
                    labels.append(1)
                    while True:
                        xn = np.random.randint(0,dis.shape[1])
                        if xn-scale>=0 and xn<dis.shape[1]-scale and x-d != xn:
                            break
                    l_ls.append(left[:,y-scale:y+1+scale,x-scale:x+1+scale])
                    r_ls.append(right[:,y-scale:y+1+scale,xn-scale:xn+1+scale])
                    ld_ls.append(left[:,y-scale:y+scale+1:2,x-scale:x+1+scale:2])
                    rd_ls.append(right[:,y-scale:y+scale+1:2,xn-scale:xn+1+scale:2])
                    labels.append(0)
def dir2img(ith):
    dis = np.round(io.imread(ith[0])/256.0).astype(int)
    left= io.imread(ith[2]) - 128.0
    left= left.swapaxes(2,1).swapaxes(1,0)
    right=io.imread(ith[1]) - 128.0
    right = right.swapaxes(2,1).swapaxes(1,0)
    return dis,left,right


# In[ ]:

scale = 6
num_epoches = 5
tot = 0
count = 0
data_list = get_data_dir(0,3)
val_list = get_data_dir(160,181)
test_list= get_data_dir(181,194)
states = {}
l_ls = []
r_ls = []
ld_ls = []
rd_ls = []
labels = []
gt = mx.nd.zeros((batch_size,),mx.gpu(3))
opt = mx.optimizer.ccSGD(learning_rate=0.00001,momentum=0.9,wd=0.00001,rescale_grad=(1.0/batch_size))
for index,key in enumerate(keys):
    if key not in data_sign:
        states[key] = opt.create_state(index,args[key])
        init(key,args[key])
        
for ith_epoche in range(num_epoches):
    for num,ith in enumerate(data_list):
        dis,left,right = dir2img(ith)
        generate_patch(left,right,dis)
        logging.info('training {}th pair img has generate {} patches'.format(num,len(labels)))
        train()

    for num,ith in enumerate(val_list):
        dis = dis,left,right = dir2img(ith)
        generate_patch(left,right,dis)
        logging.info('val {}th pair img has generate {} patches'.format(num,len(labels)))
        valdiate()
    opt.lr = opt.lr * 0.1
    mx.model.save_checkpoint('stereomatching',ith_epoche,net,args,auxs)


# In[10]:
'''
grads[]


# In[ ]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.figure(1)
plt.figure(2)
plt.figure(1)
i = 1518
plt.imshow(l_ls[i].swapaxes(0,1).swapaxes(1,2)+128)
plt.figure(2)
plt.imshow(r_ls[i].swapaxes(0,1).swapaxes(1,2)+128) 

'''