import mxnet as mx

from common.functor import Functor


class MnistModel2(Functor):

    def __init__(self):
        self.param_bounds = {
            'mdl_conv1a_nf': (6, 128),
            'mdl_conv1b_nf': (6, 128),
            'mdl_conv2a_nf': (6, 128),
            'mdl_conv2b_nf': (6, 128),
            'mdl_conv3a_nf': (6, 128),
            'mdl_conv3b_nf': (6, 128),
            'mdl_conv4a_nf': (6, 128)}

    def __call__(self,
                 num_classes=10,
                 act_type='relu',
                 mdl_conv1a_nf=12,
                 mdl_conv1b_nf=16,
                 mdl_conv2a_nf=20,
                 mdl_conv2b_nf=24,
                 mdl_conv3a_nf=28,
                 mdl_conv3b_nf=32,
                 mdl_conv4a_nf=128,
                 mdl_dropout_prob=0.1,
                 **kwargs):
        data = mx.symbol.Variable('data')
        conv1a = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv1a_nf), name='conv1a')
        act1a = mx.sym.Activation(data=conv1a, act_type=act_type, name='act1a')
        conv1b = mx.sym.Convolution(data=act1a, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv1b_nf), name='conv1b')
        act1b = mx.sym.Activation(data=conv1b, act_type=act_type, name='act1b')
        pool1 = mx.sym.Pooling(data=act1b, pool_type='max', kernel=(2, 2), stride=(2, 2), name='pool1')

        conv2a = mx.sym.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv2a_nf), name='conv2a')
        act2a = mx.sym.Activation(data=conv2a, act_type=act_type, name='act2a')
        conv2b = mx.sym.Convolution(data=act2a, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv2b_nf), name='conv2b')
        act2b = mx.sym.Activation(data=conv2b, act_type=act_type, name='act2b')
        pool2 = mx.sym.Pooling(data=act2b, pool_type='max', kernel=(2, 2), stride=(2, 2), name='pool2')

        conv3a = mx.sym.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv3a_nf), name='conv3a')
        act3a = mx.sym.Activation(data=conv3a, act_type=act_type, name='act3a')
        conv3b = mx.sym.Convolution(data=act3a, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv3b_nf), name='conv3b')
        act3b = mx.sym.Activation(data=conv3b, act_type=act_type, name='act3b')
        pool3 = mx.sym.Pooling(data=act3b, pool_type='max', kernel=(2, 2), stride=(2, 2), name='pool3')
        #dropout3 = mx.sym.Dropout(data=pool3, p=mdl_dropout_prob, name="dropout3")

        conv4a = mx.sym.Convolution(data=pool3, kernel=(3, 3), num_filter=int(mdl_conv4a_nf), name='conv4a')
        act4a = mx.sym.Activation(data=conv4a, act_type=act_type, name='act4a')
        conv4b = mx.sym.Convolution(data=act4a, kernel=(1, 1), num_filter=num_classes, name='conv4b')
        act4b = mx.sym.Activation(data=conv4b, act_type=act_type, name='act4b')

        flatten = mx.sym.Flatten(data=act4b, name='flatten')
        softmax = mx.sym.SoftmaxOutput(data=flatten, name='softmax')
        return softmax

    def show_model(self, net):
        a = mx.viz.plot_network(net, shape={"data":(1,1,28,28)}, node_attrs={"shape":'rect', "fixedsize":'false'})
        a.render("mxnet_model_graph")
