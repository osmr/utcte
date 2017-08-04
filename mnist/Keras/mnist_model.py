from utct.keras.models import Sequential
from utct.keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten

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
                 input_shape,
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

        model = Sequential()

        model.add(Convolution2D(nb_filter=int(mdl_conv1a_nf), nb_row=3, nb_col=3, activation=act_type,
                                border_mode='same', input_shape=input_shape, name='conv1a'))
        model.add(Convolution2D(nb_filter=int(mdl_conv1b_nf), nb_row=3, nb_col=3, activation=act_type,
                                border_mode='same', name='conv1b'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

        model.add(Convolution2D(nb_filter=int(mdl_conv2a_nf), nb_row=3, nb_col=3, activation=act_type,
                                border_mode='same', name='conv2a'))
        model.add(Dropout(p=mdl_drop2a_p, name='drop2a'))
        model.add(Convolution2D(nb_filter=int(mdl_conv2b_nf), nb_row=3, nb_col=3, activation=act_type,
                                border_mode='same', name='conv2b'))
        model.add(Dropout(p=mdl_drop2b_p, name='drop2b'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(output_dim=int(mdl_fc1_nh), activation=act_type, name='fc1'))
        model.add(Dropout(p=mdl_drop3_p, name='drop3'))
        model.add(Dense(output_dim=num_classes, activation='softmax', name='fc2'))

        return model
