import numpy as np
from cntk.layers import Convolution, MaxPooling, Dropout, Dense
from cntk.initializer import glorot_uniform
from cntk.ops import input_variable, relu, softmax

from common.functor import Functor


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
        self.img_h = 28
        self.img_w = 28
        self.n_dim = 10

    def __call__(self,
                 num_classes=10,
                 act_type=relu,
                 mdl_conv1a_nf=40,
                 mdl_conv1b_nf=60,
                 mdl_conv2a_nf=50,
                 mdl_conv2b_nf=75,
                 mdl_fc1_nh=75,
                 mdl_drop2a_p=0.033,
                 mdl_drop2b_p=0.097,
                 mdl_drop3_p=0.412,
                 **kwargs):

        input_var = input_variable((1, self.img_h, self.img_w), np.float32)
        label_var = input_variable((self.n_dim), np.float32)

        conv1a = Convolution(filter_shape=(3, 3), num_filters=int(mdl_conv1a_nf), activation=act_type, init=glorot_uniform(), pad=True, name='conv1a')(input_var)
        conv1b = Convolution(filter_shape=(3, 3), num_filters=int(mdl_conv1b_nf), activation=act_type, init=glorot_uniform(), pad=True, name='conv1b')(conv1a)
        pool1 = MaxPooling(filter_shape=(2, 2), strides=(2, 2), name='pool1')(conv1b)

        conv2a = Convolution(filter_shape=(3, 3), num_filters=int(mdl_conv2a_nf), activation=act_type, init=glorot_uniform(), pad=True, name='conv2a')(pool1)
        drop2a = Dropout(prob=mdl_drop2a_p, name="drop2a")(conv2a)
        conv2b = Convolution(filter_shape=(3, 3), num_filters=int(mdl_conv2b_nf), activation=act_type, init=glorot_uniform(), pad=True, name='conv2b')(drop2a)
        drop2b = Dropout(prob=mdl_drop2a_p, name="drop2a")(conv2b)
        pool2 = MaxPooling(filter_shape=(2, 2), strides=(2, 2), name='pool2')(drop2b)

        fc1 = Dense(shape=int(mdl_fc1_nh), init=glorot_uniform(), activation=act_type, name='fc1')(pool2)
        drop3 = Dropout(prob=mdl_drop3_p, name="drop3")(fc1)
        #fc2 = Dense(shape=num_classes, init=glorot_uniform(), activation=softmax, name='fc2')(drop3)
        fc2 = Dense(shape=num_classes, init=glorot_uniform(), activation=None, name='fc2')(drop3)

        return input_var, label_var, fc2
