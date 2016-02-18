import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import logging
import math
from skimage import io
from sklearn import utils
import argparse
from model import get_network
from random import randint
from random import shuffle
logging.basicConfig( 
                    level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt ='%a, %d %b %Y %H:%M:%S'
                    )
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
        left = io.imread(self.data[ith][1]) - 128.0
        left = left.swapaxes(2,1).swapaxes(1,0)
        right= io.imread(self.data[ith][2]) - 128.0
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
            imgR = './colored_1/'+dir_name+'_10.png'.format(num)
            self.data.append((gt,imgL,imgR))
        shuffle(self.data)

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
                        self.l_ls.append( left[:,y-scale:y+1+scale,    x-scale:x+1+scale])
                        self.r_ls.append(right[:,y-scale:y+1+scale,xn-scale:xn+1+scale])
                        self.ld_ls.append(left[:,y-scale:y+scale+1:2,x-scale:x+1+scale:2])
                        self.rd_ls.append(right[:,y-scale:y+scale+1:2,xn-scale:xn+1+scale:2])
                        self.labels.append(0) 
                        self.volume +=2      

    def load_data(self,batch_size=1235):
        #args['left_downsample'][:batch_size] = mx.nd.array(self.ld_ls[:batch_size])
        #args['right_downsample'][:batch_size]= mx.nd.array(self.rd_ls[:batch_size])
        args['left'][:batch_size][:] = np.asarray(self.l_ls[:batch_size])
        args['right'][:batch_size][:]= np.asarray(self.r_ls[:batch_size])
        args['gt'][:] = mx.nd.array(self.labels[:batch_size],ctx) 
        self.volume -= batch_size  

        del self.l_ls[:batch_size]
        del self.rd_ls[:batch_size]
        del self.ld_ls[:batch_size]
        del self.r_ls[:batch_size]
        del self.labels[:batch_size]

def init(key,weight):
    if 'bias' in key:
        weight[:] = 0
    elif key not in data_sign:
        weight[:] = mx.random.uniform(-0.007,0.007,weight.shape)    

def train(data_iter):
    count = 0
    tot   = 0
    
    while data_iter.volume >= batch_size:  
        data_iter.load_data() 
        count +=1
        executor.forward(is_train=True)
        output = executor.outputs[2]
        grad = 2*(output-args['gt'])
        loss = (mx.nd.square(output-args['gt']).asnumpy())
        tot += loss.sum()
        loss = loss.mean()
        tmp = output.asnumpy()
        acc = tmp[args['gt'].asnumpy()==1].mean()
        err = tmp[args['gt'].asnumpy()==0].mean()
        idx = randint(30,1000)
        l_p = args['left'].asnumpy()[idx].swapaxes(0,1).swapaxes(1,2) + 128
        r_p = args['right'].asnumpy()[idx].swapaxes(0,1).swapaxes(1,2) + 128
        result = (output.asnumpy()[idx],args['gt'].asnumpy()[idx])
        plt.figure()
        p1 = plt.subplot(211)
        p2 = plt.subplot(212)
        p1.imshow(l_p)
        p2.imshow(r_p)
        #plt.show()
        #print args['bias1_blue'].asnumpy()
        global no
        plt.savefig('./result/%d_%d_%.5f.jpg' % (no,result[1],result[0]))
        no+=1
        plt.close()
        logging.info("training: {}th pair img:{}th iteration square loss:{} acc:{} err:{} >:{} lr:{}".format(num,count,loss,acc,err,acc-err,opt.lr))
        executor.backward([mx.nd.zeros((batch_size,33800),ctx=ctx),mx.nd.zeros((batch_size,33800),ctx=ctx),grad])
        #print grads['l3_blue'].asnumpy().max()
        for index,key in enumerate(keys):
            if key not in data_sign:
                opt.update(index,args[key],grads[key],states[key])
                grads[key][:] = np.zeros(grads[key].shape)
    logging.info("the {}th img avg square loss :{}".format(num,tot/float(count*batch_size)))

def valdiate(val_iter):
    count = 0
    tot   = 0
    while val_iter.volume >= batch_size:  
        val_iter.load_data() 
        count +=1
        executor.forward(is_train=False)
        output = executor.outputs[2]
        loss = (mx.nd.square(output-args['gt']).asnumpy())
        tot += loss.sum()
        loss = loss.mean()
        tmp = output.asnumpy()
        acc = tmp[args['gt'].asnumpy()==1].mean()
        err = tmp[args['gt'].asnumpy()==0].mean()
        logging.info("val: {}th pair img:{}th iteration square loss:{} acc:{} err:{} lr:{}".format(num,count,loss,acc,err,opt.lr))
    logging.info("the {}th img avg square loss :{}".format(num,tot/float(count*batch_size)))

def predict(left,right):
    dis_pred = np.zeros((dis.shape[0],batch_size))
    values   = np.zeros((dis.shape[0],batch_size))
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
    
    no = 0

    data_sign = ['left','right','left_downsample','right_downsample','label','LinearRegression_label','gt']
    
    if cmd.con == -1:
        lr = 0.0001 #* (0.1**(cmd.con+1))
        net = get_network()
        executor = net.simple_bind(ctx=ctx,grad_req='add',left = s1,right= s1)
        keys  = net.list_arguments()
        grads = dict(zip(net.list_arguments(),executor.grad_arrays))
        args  = dict(zip(keys,executor.arg_arrays))
        auxs  = dict(zip(keys,executor.arg_arrays))
        args['gt'] = mx.nd.zeros((batch_size,),ctx)
        logging.info("complete network architecture design")
    else:
        lr = 0.000001 #* (0.1**(cmd.con+2))
        net,args,aux = mx.model.load_checkpoint('stereo',cmd.con)
        keys = net.list_arguments()
        executor = net.simple_bind(ctx=ctx,grad_req='add',left = s1,right= s1)
        for key in executor.arg_dict:
            if key not in  data_sign:
                executor.arg_dict[key][:] = args[key]
            else:
                executor.arg_dict[key][:] = mx.nd.zeros((executor.arg_dict[key].shape),ctx)
        
        grads = dict(zip(net.list_arguments(),executor.grad_arrays))
        args  = dict(zip(keys,executor.arg_arrays))
        auxs  = dict(zip(keys,executor.arg_arrays))
        args['gt'] = mx.nd.zeros((batch_size,),ctx)
        logging.info("load the paramaters and net")
    
    scale = 6
    num_epoches = 3
    count = 0
    
    data_iter =  dataiter(0,175)
    val_iter  =  dataiter(175,180)
    test_iter =  dataiter(180,194)
    states    =  {}

    #init args
    opt = mx.optimizer.ccSGD(learning_rate=lr,momentum=0.9,wd=0.000003,rescale_grad=(1.0/batch_size))
    for index,key in enumerate(keys):
        if key not in data_sign:
            states[key] = opt.create_state(index,args[key])
            if cmd.con == -1 :
                init(key,args[key])
    # train + val  
    index = 0      
    for ith_epoche in range(num_epoches):
        for num in range(data_iter.len):
            data_iter.produce_patch(num)
            logging.info('training {}th pair img has generate {} patches'.format(num,data_iter.volume))
            #if num % 5 == 0 and num>0:
                #opt.lr *= 0.3
            train(data_iter)
            mx.model.save_checkpoint('stereo',index,net,args,auxs)
            index+=1
        opt.lr *=0.3
        mx.model.save_checkpoint('stereomatching',ith_epoche+cmd.con+1,net,args,auxs)
        if cmd.val == 1:
            for num in range(val_iter.len):
                val_iter.produce_patch(num)
                logging.info('validating {}th pair img has generate {} patches'.format(num,len(val_iter.labels)))
                valdiate(val_iter)
        opt.lr = opt.lr * 0.1

#-----------------------predict-------------------------#

    for num,ith in enumerate(test_iter.data):
        dis   = np.round(io.imread(ith[0])/256.0).astype(int)
        left  = io.imread(ith[2]) - 128.0
        left  = left.swapaxes(2,1).swapaxes(1,0)
        right = io.imread(ith[1]) - 128.0
        right = right.swapaxes(2,1).swapaxes(1,0)
        pred,values = predict(left,right)
        io.imsave('./result/{}.png'.format(num),pred[:dis.shape[0],:dis.shape[1]])
    logging.info('Success!!')