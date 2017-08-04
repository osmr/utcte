import mxnet as mx

from utct.common.functor import Functor


class MnistModel(Functor):

    def __init__(self):
        super(MnistModel, self).__init__()
        #self.param_bounds = {
        #    #'mdl_conv1a_nf': (6, 128),
        #    #'mdl_conv1b_nf': (6, 128),
        #    #'mdl_conv2a_nf': (6, 128),
        #    #'mdl_conv2b_nf': (6, 128),
        #    #'mdl_fc1_nh': (10, 500),
        #    'mdl_drop2a_p': (0.0, 0.25),
        #    'mdl_drop2b_p': (0.0, 0.25),
        #    'mdl_drop3_p': (0.0, 0.50)}

    def __call__(self,
                 num_classes=10,
                 act_type='relu',
                 mdl_conv1a_nf=40,
                 mdl_conv1b_nf=60,
                 mdl_conv2a_nf=50,
                 mdl_conv2b_nf=75,
                 mdl_fc1_nh=75,
                 mdl_drop2a_p=0.033,
                 mdl_drop2b_p=0.097,
                 mdl_drop3_p=0.412,
                 **kwargs):
        data = mx.symbol.Variable('data')
        conv1a = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv1a_nf), name='conv1a')
        act1a = mx.sym.Activation(data=conv1a, act_type=act_type, name='act1a')
        conv1b = mx.sym.Convolution(data=act1a, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv1b_nf), name='conv1b')
        act1b = mx.sym.Activation(data=conv1b, act_type=act_type, name='act1b')
        pool1 = mx.sym.Pooling(data=act1b, pool_type='max', kernel=(2, 2), stride=(2, 2), name='pool1')

        conv2a = mx.sym.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv2a_nf), name='conv2a')
        act2a = mx.sym.Activation(data=conv2a, act_type=act_type, name='act2a')
        drop2a = mx.sym.Dropout(data=act2a, p=mdl_drop2a_p, name="drop2a")
        conv2b = mx.sym.Convolution(data=drop2a, kernel=(3, 3), pad=(1, 1), num_filter=int(mdl_conv2b_nf), name='conv2b')
        act2b = mx.sym.Activation(data=conv2b, act_type=act_type, name='act2b')
        drop2b = mx.sym.Dropout(data=act2b, p=mdl_drop2b_p, name="drop2b")
        pool2 = mx.sym.Pooling(data=drop2b, pool_type='max', kernel=(2, 2), stride=(2, 2), name='pool2')

        flatten = mx.sym.Flatten(data=pool2, name='flatten')
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=int(mdl_fc1_nh), name='fc1')
        act3 = mx.sym.Activation(data=fc1, act_type=act_type, name='act3')
        drop3 = mx.sym.Dropout(data=act3, p=mdl_drop3_p, name="drop3")

        fc2 = mx.sym.FullyConnected(data=drop3, num_hidden=num_classes, name='fc2')
        softmax = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
        return softmax

    def show_model(self, net):
        a = mx.viz.plot_network(net, shape={"data":(1,1,28,28)}, node_attrs={"shape":'rect', "fixedsize":'false'})
        a.render("mxnet_model_graph")
