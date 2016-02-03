import mxnet as mx
import numpy as np
import logging
import math
from skimage import io
from sklearn import utils
import argparse

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt ='%a, %d %b %Y %H:%M:%S')
                #,
               # filename ='train_log.log',
               # filemode='w')

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

class dataiter:
    def __init__(self,low,high):
        self.get_data_dir(low,high)
        self.l_ls = []
        self.r_ls = []
        self.ld_ls = []
        self.rd_ls = []
        self.labels = []
        self.len  = high-low
        self.volume = 0
    def produce_patch(self,ith):
        #load img dir
        dis  = np.round(io.imread(self.data[ith][0])/256.0).astype(int)
        left = io.imread(self.data[ith][2]) - 128.0
        left = left.swapaxes(2,1).swapaxes(1,0)
        right= io.imread(self.data[ith][1]) - 128.0
        right= right.swapaxes(2,1).swapaxes(1,0)
        self.generate_patch(left,right,dis)  
        utils.shuffle(self.l_ls,self.r_ls,self.ld_ls,self.rd_ls,self.labels,random_state=0)

    def get_data_dir(self,low,high):
        self.data = []
        for num in range(low,high):
            dir_name = '000{}'.format(num)
            if len(dir_name) ==4 :
                dir_name = '00'+dir_name
            elif len(dir_name) == 5:
                dir_name = '0'+dir_name
            gt = './disp_noc/'+dir_name+'_10.png'.format(num)
            imgL = './colored_0/'+dir_name+'_10.png'.format(num)
            imgR ='./colored_0/'+dir_name+'_11.png'.format(num)
            self.data.append((gt,imgL,imgR))

    def generate_patch(self,left,right,dis):
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
                            xn = np.random.randint(scale,dis.shape[1]-scale)
                            if xn<dis.shape[1]-scale and x-d != xn:
                                break
                        self.l_ls.append( left[:,y-scale:y+1+scale,  x-scale:x+1+scale])
                        self.r_ls.append(right[:,y-scale:y+1+scale,xn-scale:xn+1+scale])
                        self.ld_ls.append(left[:,y-scale:y+scale+1:2,x-scale:x+1+scale:2])
                        self.rd_ls.append(right[:,y-scale:y+scale+1:2,xn-scale:xn+1+scale:2])
                        self.labels.append(0) 
                        self.volume +=2      

    def load_data(self,batch_size=1235):
        args['left'][:batch_size][:] = np.asarray(self.l_ls[:batch_size])
        del self.l_ls[:batch_size]
        args['right'][:batch_size][:]= np.asarray(self.r_ls[:batch_size])
        del self.r_ls[:batch_size]
        args['gt'][:] = mx.nd.array(self.labels[:batch_size],ctx)   
        del self.labels[:batch_size]
        self.volume -= batch_size
 #      args['left_downsample'][:batch_size] = mx.nd.array(self.ld_ls[:batch_size])
        del self.ld_ls[:batch_size]
 #      args['right_downsample'][:batch_size]= mx.nd.array(self.rd_ls[:batch_size])
        del self.rd_ls[:batch_size]
    

def init(key,weight):
    if 'bias' in key:
        weight[:] = 0
    elif key not in data_sign:
        weight[:] = mx.random.uniform(-0.007,0.007,weight.shape)    

def train():
    count = 0
    tot = 0
    while data_iter.volume >= batch_size:  
        data_iter.load_data() 
        count +=1
        executor.forward(is_train=True)
        output = executor.outputs[2]
        grad = 2*(output-args['gt'])
        loss = (mx.nd.square(output-args['gt']).asnumpy()).mean()
        tot += (mx.nd.square(output-args['gt']).asnumpy()).sum()
        logging.info("training: {}th pair img:{}th iteration square loss:{}".format(num,count,loss))
        executor.backward([mx.nd.zeros((batch_size,33800),ctx=ctx),mx.nd.zeros((batch_size,33800),ctx=ctx),grad])
        for index,key in enumerate(keys):
            if key not in data_sign:
                opt.update(index,args[key],grads[key],states[key])
    logging.info("the {}th img avg square loss :{}".format(num,tot/float(count*batch_size)))

def valdiate():
    count = 0
    tot   = 0
    while val_iter.volume >= batch_size:  
        val_iter.load_data() 
        count +=1
        executor.forward(is_train=False)
        output = executor.outputs[2]
        loss = (mx.nd.square(output-args['gt']).asnumpy()).mean()
        tot += (mx.nd.square(output-args['gt']).asnumpy()).sum()
        logging.info("validate: {}th pair img:{}th iteration square loss:{}".format(num,count,loss))
    logging.info("the {}th img avg square loss :{}".format(num,tot/float(count*batch_size)))

def predict(left,right):
    dis_pred = np.zeros((dis.shape[0]-2*scale,batch_size))
    values   = np.zeros((dis.shape[0]-2*scale,batch_size))
    for y in xrange(scale,dis.shape[0]-scale):
        for x in xrange(scale,dis.shape[1]-scale):
            test_iter.l_ls.append( left[:,y-scale:y+1+scale,x-scale:x+1+scale])
            test_iter.r_ls.append(right[:,y-scale:y+1+scale,x-scale:x+1+scale])
            test_iter.ld_ls.append(left[:,y-scale:y+1+scale:2,x-scale:x+1+scale:2])
            test_iter.rd_ls.append(right[:,y-scale:y+1+scale:2,x-scale:x+1+scale:2])
            test_iter.volume += 1
        test_iter.load_data(dis.shape[1]-2*scale)
        executor.forward(is_train=False)  
        out1 = executor.outputs[0]
        out2 = mx.nd.array(executor.outputs[1].asnumpy().swapaxes(0,1),ctx)
        x = mx.nd.array(range(batch_size),ctx)
        dis_pred[y,:]  =  (x -  mx.nd.argmax_channel(mx.nd.dot(out1,out2))).asnumpy()
        values[y,:] = mx.nd.argmax_channel(mx.nd.dot(out1,out2)).asnumpy()
    return dis_pred,values

#--------------------------------------------------main------------------------------------------------#


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--c',action='store',dest='con',type=int)
    parser.add_argument('--v',action='store',dest='val',type=int)
    cmd = parser.parse_args()
    batch_size = 1235
    s1 = (batch_size,3,13,13)
    s2 = (batch_size,3,7,7)
    ctx = mx.gpu(3) 
    data_sign = ['left','right','left_downsample','right_downsample','label','LinearRegression_label','gt']
    if cmd.con == -1:
        data_sign = ['left','right','left_downsample','right_downsample','label','LinearRegression_label','gt']
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
        relu1_1_blue = mx.symbol.Activation(data=conv1_1_blue, act_type="relu")
        conv1_2_blue = mx.sym.Convolution(data=right,weight=weight1_blue,bias =b1_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
        relu1_2_blue = mx.symbol.Activation(data=conv1_2_blue, act_type="relu")
        conv2_1_blue = mx.sym.Convolution(data=relu1_1_blue,weight=weight2_blue,bias = b2_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
        relu2_1_blue = mx.symbol.Activation(data=conv2_1_blue, act_type="relu")
        conv2_2_blue = mx.sym.Convolution(data=relu1_2_blue,weight=weight2_blue,bias = b2_blue,kernel=(3,3),pad=(1,1),num_filter = 32)
        relu2_2_blue = mx.symbol.Activation(data=conv2_2_blue, act_type="relu")
        conv3_1_blue = mx.sym.Convolution(data=relu2_1_blue,weight=weight3_blue,bias = b3_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
        relu3_1_blue = mx.symbol.Activation(data=conv3_1_blue, act_type="relu")
        conv3_2_blue = mx.sym.Convolution(data=relu2_2_blue,weight=weight3_blue,bias = b3_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
        relu3_2_blue = mx.symbol.Activation(data=conv3_2_blue, act_type="relu")
        conv4_1_blue = mx.sym.Convolution(data=relu3_1_blue,weight=weight4_blue,bias = b4_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
        relu4_1_blue = mx.symbol.Activation(data=conv4_1_blue, act_type="relu")
        conv4_2_blue = mx.sym.Convolution(data=relu3_2_blue,weight=weight4_blue,bias = b4_blue,kernel=(5,5),pad=(2,2),num_filter = 200)
        relu4_2_blue = mx.symbol.Activation(data=conv4_2_blue, act_type="relu")
      
        conv1_1_red = mx.sym.Convolution(data=leftdownsample,weight=weight1_red,bias = b1_red,kernel=(3,3),pad=(1,1),num_filter = 32)
        conv1_2_red = mx.sym.Convolution(data=rightdownsample,weight=weight1_red,bias = b1_red,kernel=(3,3),pad=(1,1),num_filter =32)
        conv2_1_red = mx.sym.Convolution(data=conv1_1_red,weight=weight2_red,bias = b2_red,kernel=(3,3),pad=(1,1),num_filter = 32)
        conv2_2_red = mx.sym.Convolution(data=conv1_2_red,weight=weight2_red,bias = b2_red,kernel=(3,3),pad=(1,1),num_filter = 32)
        conv3_1_red = mx.sym.Convolution(data=conv2_1_red,weight=weight3_red,bias = b3_red,kernel=(5,5),pad=(2,2),num_filter = 200)
        conv3_2_red = mx.sym.Convolution(data=conv2_2_red,weight=weight3_red,bias = b3_red,kernel=(5,5),pad=(2,2),num_filter = 200)
        conv4_1_red = mx.sym.Convolution(data=conv3_1_red,weight=weight4_red,bias = b4_red,kernel=(5,5),pad=(2,2),num_filter = 200)
        conv4_2_red = mx.sym.Convolution(data=conv3_2_red,weight=weight4_red,bias = b4_red,kernel=(5,5),pad=(2,2),num_filter = 200)

        f_blue1 = mx.sym.Flatten(data = relu4_1_blue)
        f_blue2 = mx.sym.Flatten(data = relu4_2_blue)
        f_red1  = mx.sym.Flatten(data = conv4_1_red)
        f_red2  = mx.sym.Flatten(data = conv4_2_red)
        s = mx.sym.Dotproduct(   data1= f_blue1, data2 = f_blue2 )
        net = mx.sym.Group([f_blue1,f_blue2,s])

        args_shape,out_shape,aux_shape = net.infer_shape(left=s1,right=s1)
        args_shape = dict(zip(net.list_arguments(),args_shape))
        executor = net.simple_bind(ctx=ctx,grad_req='write',left = s1,right= s1)
        keys = net.list_arguments()
        args = executor.arg_arrays
        grads = dict(zip(net.list_arguments(),executor.grad_arrays))
        args = dict(zip(keys,args))
        auxs = dict(zip(keys,executor.arg_arrays))
        lr = 0.000001
        logging.info("complete network architecture design")
        args['gt'] = mx.nd.zeros((batch_size,),ctx)
    else:
        net,args,aux =  mx.model.load_checkpoint('stereomatching',cmd.con)
        keys = net.list_arguments()
        executor = net.simple_bind(ctx=ctx,grad_req='write',left = s1,right= s1)
        grads = dict(zip(net.list_arguments(),executor.grad_arrays))
        lr = 0.000001 #* (0.1**(cmd.con+1))
        args['gt'] = mx.nd.zeros((batch_size,),ctx)
        for key in executor.arg_dict:
            if key not in  data_sign:
                executor.arg_dict[key][:] = args[key]
            else :
                executor.arg_dict[key][:] = mx.nd.zeros((executor.arg_dict[key].shape),ctx)
        logging.info("Have load the paramaters and net")
        

    scale = 6
    num_epoches = 1
    count = 0
    data_iter =  dataiter(0,10)
    val_iter  =  dataiter(175,180)
    test_iter =  dataiter(180,194)
    states    =  {}
    
    opt = mx.optimizer.ccSGD(learning_rate=lr,momentum=0.9,wd=0.00001,rescale_grad=(1.0/batch_size))


    for index,key in enumerate(keys):
        if key not in data_sign:
            states[key] = opt.create_state(index,args[key])
            if cmd.con == -1 :
                init(key,args[key])
    # train + val        
    for ith_epoche in range(num_epoches):
        for num in range(data_iter.len):
            data_iter.produce_patch(num)
            logging.info('training {}th pair img has generate {} patches'.format(num,data_iter.volume))
            train()
        mx.model.save_checkpoint('stereomatching',ith_epoche+cmd.con+1,net,args,auxs)
        if cmd.val == 1:
            for num in range(val_iter.len):
                val_iter.produce_patch(num)
                logging.info('validating {}th pair img has generate {} patches'.format(num,len(val_iter.labels)))
                valdiate()
        opt.lr = opt.lr * 0.1

#-----------------------predict-------------------------#

    for num,ith in enumerate(test_iter.data):
        dis = np.round(io.imread(ith[0])/256.0).astype(int)
        left= io.imread(ith[2]) - 128.0
        left= left.swapaxes(2,1).swapaxes(1,0)
        right=io.imread(ith[1]) - 128.0
        right = right.swapaxes(2,1).swapaxes(1,0)
        pred,values = predict(left,right)
        io.imsave('./result/{}.png'.format(num),pred[:dis.shape[0],:dis.shape[1]])

    logging.info('Success!!')