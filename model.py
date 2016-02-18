import mxnet as mx
def get_network():
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
        #dot = Dot()
        #s = dot(x = f_blue1 ,y =f_blue2,name='dot_product1') #+ w2*dot(x = f_red1,y = f_red2,name='dot_product2')
        s = mx.sym.Dotproduct(   data1 = f_blue1, data2 = f_blue2 )
        net = mx.sym.Group([f_blue1,f_blue2,s])
        return net

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
