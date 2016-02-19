# -*- coding:utf-8 -*-  
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
from model import get_network,dataiter
from random import randint
from random import shuffle
from collections import namedtuple

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt ='%d %H:%M:%S')

def init(key,weight):
    if 'bias' in key:
        weight[:] = 0
    elif key not in data_sign:
        weight[:] = mx.random.uniform(-0.007,0.007,weight.shape)    

def draw_patch(img_idx):
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
    shuffle(img_dir)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  
    parser.add_argument('--continue',action='store',dest='con',type=int)
    parser.add_argument('--lr',action='store',dest='lr',type=float)
    cmd = parser.parse_args()
    #cmd.con 不是指epoch，是指第几个轮，200个batch 为1轮 。 kitty dataset 跑一次epoch 需要3 小时

    lr = cmd.lr
    batch_size = 1235
    s1 = (batch_size,3,13,13)
    s2 = (batch_size,3,7,7)
    ctx = mx.gpu(3) 
    data_sign = ['left','right','left_downsample','right_downsample','label','LinearRegression_label','gt']
    
    if cmd.con == -1:
        #重新训练
        net = get_network('not fully')
        executor = net.simple_bind(ctx=ctx,grad_req='add',left = s1,right= s1)
        keys  = net.list_arguments()
        grads = dict(zip(net.list_arguments(),executor.grad_arrays))
        args  = dict(zip(keys,executor.arg_arrays))
        auxs  = dict(zip(keys,executor.arg_arrays))
        args['gt'] = mx.nd.zeros((batch_size,),ctx)
        logging.info("complete network architecture design")
    
    else:
        #继续之前的训练
        net,executor = load_model('stereo',cmd.con,s1,'not fully',ctx)
        keys = net.list_arguments()
        grads = dict(zip(keys,executor.grad_arrays))
        args  = dict(zip(keys,executor.arg_arrays))
        auxs  = dict(zip(keys,executor.arg_arrays))
        args['gt'] = mx.nd.zeros((batch_size,),ctx)
        logging.info("load the paramaters and net")


    scale = 6
    num_epoches = 1
    train_iter =  dataiter(get_kitty_data_dir(0,175),batch_size,ctx,'train')
    val_iter   =  dataiter(get_kitty_data_dir(175,180),batch_size,ctx,'valdiate')
    states     =  {}

    #init args and optimizer
    opt = mx.optimizer.ccSGD(learning_rate=lr,momentum=0.9,wd=0.000003,rescale_grad=(1.0/batch_size))
    for index,key in enumerate(keys):
        if key not in data_sign:
            states[key] = opt.create_state(index,args[key])
            if cmd.con == -1 :
                init(key,args[key])
    
    # train + validate 
    for ith_epoche in range(num_epoches):
        train_iter.reset()
        val_iter.reset()
        train_loss = 0.0
        val_loss = 0.0
        nbatch = 0
        loss_of_30 = 0.0
        #train
        for dbatch in train_iter:
            #load data
            args['left'][:]  = dbatch.data[0]
            args['right'][:] = dbatch.data[1]
            args['gt'][:] = dbatch.label
            nbatch += 1
            
            #forward 
            executor.forward(is_train=True)
            output = executor.outputs[2]
            grad = 2*(output-args['gt'])
            draw_patch(train_iter.img_idx)
            #calc loss
            loss = (mx.nd.square(output-args['gt']).asnumpy()).mean()
            train_loss += loss
            loss_of_30 += loss
            tmp = output.asnumpy()
            acc = tmp[args['gt'].asnumpy()==1].mean()
            err = tmp[args['gt'].asnumpy()==0].mean()

            #logging.info("training: {}th pair img:{}th l2 loss:{} acc:{} err:{} >:{} lr:{}".format(nbatch,train_iter.img_idx,loss,acc,err,acc-err,opt.lr))
            #30轮的平均loss
            if nbatch % 30 == 0:
                logging.info("mean loss of 30 batches: {} ".format(loss_of_30/30.0))
                loss_of_30 = 0.0
                draw_patch(train_iter.img_idx)
        
            #update args
            executor.backward([mx.nd.zeros((batch_size,200),ctx=ctx),mx.nd.zeros((batch_size,200),ctx=ctx),grad])
            for index,key in enumerate(keys):
                if key not in data_sign:
                    opt.update(index,args[key],grads[key],states[key])
                    grads[key][:] = np.zeros(grads[key].shape)
            
            if nbatch % 200==0:
                cmd.con+=1
                mx.model.save_checkpoint('stereo',cmd.con,net,args,auxs)

        logging.info('training: ith_epoche :{} mean loss:{}'.format(ith_epoche,train_loss/float(nbatch)))
        
        #eval
        nbatch = 0.0
        for dbatch in val_iter:
            args['left'][:]  = dbatch.data[0]
            args['right'][:] = dbatch.data[1]
            args['gt'][:] = dbatch.label
            nbatch += 1
            executor.forward(is_train=False)
            loss = (mx.nd.square(executor.outputs[2]-args['gt']).asnumpy()).mean()
            val_loss += loss
            logging.info('eval:{}th pair img:{}th l2 loss:{}'.format(nbatch,val_iter.img_idx,loss))
        logging.info('eval : ith_epoch :{} mean loss:{}'.format(ith_epoche,val_loss/float(nbatch)))
    logging.info('Successful!')