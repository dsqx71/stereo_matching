# -*- coding:utf-8 -*-  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import logging
import argparse
from model import get_network,dataiter
from util import get_kitty_data_dir,load_model,draw_patch

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt ='%d %H:%M:%S')

def init(key,weight):
    if 'bias' in key:
        weight[:] = 0
    elif key not in data_sign:
        weight[:] = mx.random.uniform(-0.007,0.007,weight.shape)    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()  
    parser.add_argument('--continue',action='store',dest='con',type=int)
    parser.add_argument('--lr',action='store',dest='lr',type=float)
    parser.add_argument('--l',action='store',dest='low',type=int)
    parser.add_argument('--h',action='store',dest='high',type=int)
    cmd = parser.parse_args()

    #cmd.con 不是指epoch，是指第几个轮，200个batch 为1轮 。 kitty dataset 跑一次epoch 需要3 小时

    lr = cmd.lr
    batch_size = 10000
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
    num_epoches = 10
    train_iter =  dataiter(get_kitty_data_dir(0,175),batch_size,ctx,cmd.low,cmd.high,'train')
    val_iter   =  dataiter(get_kitty_data_dir(175,180),batch_size,ctx,cmd.low,cmd.high,'valdiate')
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
            draw_patch(args,executor,train_iter.img_idx)
            #calc loss
            loss = (mx.nd.square(output-args['gt']).asnumpy()).mean()
            train_loss += loss
            loss_of_30 += loss
            tmp = output.asnumpy()
            acc = tmp[args['gt'].asnumpy()==1].mean()
            err = tmp[args['gt'].asnumpy()==0].mean()

            logging.info("training: {}th pair img:{}th l2 loss:{} acc:{} err:{} >:{} lr:{}".format(nbatch,train_iter.img_idx,loss,acc,err,acc-err,opt.lr))
            #30轮的平均loss
            if nbatch % 30 == 0:
                logging.info("mean loss of 30 batches: {} ".format(loss_of_30/30.0))
                loss_of_30 = 0.0
                #draw_patch(train_iter.img_idx)
        
            #update args
            executor.backward([mx.nd.zeros((batch_size,200),ctx=ctx),mx.nd.zeros((batch_size,200),ctx=ctx),grad])
            for index,key in enumerate(keys):
                if key not in data_sign:
                    opt.update(index,args[key],grads[key],states[key])
                    grads[key][:] = np.zeros(grads[key].shape)
            
            if nbatch % 20==0:
                cmd.con = (cmd.con + 1) % 2000
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