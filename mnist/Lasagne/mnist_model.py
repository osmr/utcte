import lasagne

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
                 input_var=None,
                 num_classes=10,
                 nl_type=lasagne.nonlinearities.rectify,
                 mdl_conv1a_nf=40,
                 mdl_conv1b_nf=60,
                 mdl_conv2a_nf=50,
                 mdl_conv2b_nf=75,
                 mdl_fc1_nh=75,
                 mdl_drop2a_p=0.033,
                 mdl_drop2b_p=0.097,
                 mdl_drop3_p=0.412,
                 **kwargs):

        data = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var, name='data')
        conv1a = lasagne.layers.Conv2DLayer(incoming=data, num_filters=int(mdl_conv1a_nf), filter_size=(3, 3),
                                            pad=(1, 1), nonlinearity=nl_type, name='conv1a')
        conv1b = lasagne.layers.Conv2DLayer(incoming=conv1a, num_filters=int(mdl_conv1b_nf), filter_size=(3, 3),
                                            pad=(1, 1), nonlinearity=nl_type, name='conv1b')
        pool1 = lasagne.layers.MaxPool2DLayer(incoming=conv1b, pool_size=(2, 2), name='pool1')

        conv2a = lasagne.layers.Conv2DLayer(incoming=pool1, num_filters=int(mdl_conv2a_nf), filter_size=(3, 3),
                                            pad=(1, 1), nonlinearity=nl_type, name='conv2a')
        drop2a = lasagne.layers.DropoutLayer(incoming=conv2a, p=mdl_drop2a_p, name="drop2a")
        conv2b = lasagne.layers.Conv2DLayer(incoming=drop2a, num_filters=int(mdl_conv2b_nf), filter_size=(3, 3),
                                            pad=(1, 1), nonlinearity=nl_type, name='conv2b')
        drop2b = lasagne.layers.DropoutLayer(incoming=conv2b, p=mdl_drop2b_p, name="drop2b")
        pool2 = lasagne.layers.MaxPool2DLayer(incoming=drop2b, pool_size=(2, 2), name='pool2')

        fc1 = lasagne.layers.DenseLayer(incoming=pool2, num_units=int(mdl_fc1_nh), nonlinearity=nl_type, name='fc1')
        drop3 = lasagne.layers.DropoutLayer(incoming=fc1, p=mdl_drop3_p, name="drop3")

        fc2 = lasagne.layers.DenseLayer(incoming=drop3, num_units=num_classes,
                                        nonlinearity=lasagne.nonlinearities.softmax, name='fc2')
        return fc2
