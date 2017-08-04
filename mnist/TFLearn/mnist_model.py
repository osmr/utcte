import tensorflow as tf
import tflearn

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
        self.img_h = 28
        self.img_w = 28

    def __call__(self,
                 optimizer,
                 data_source,
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
                 is_training=True,
                 **kwargs):

        x = tflearn.input_data(
            shape=[None, self.img_h, self.img_w, 1],
            data_augmentation=data_source.img_aug,
            name='X')
        y = tf.placeholder(tf.float32, shape=(None, num_classes), name='YY')
        y_est = self._calc_y_est(x,
                                 num_classes,
                                 act_type,
                                 mdl_conv1a_nf,
                                 mdl_conv1b_nf,
                                 mdl_conv2a_nf,
                                 mdl_conv2b_nf,
                                 mdl_fc1_nh,
                                 mdl_drop2a_p if is_training else 0.0,
                                 mdl_drop2b_p if is_training else 0.0,
                                 mdl_drop3_p if is_training else 0.0)
        net = tflearn.regression(y_est, y,
                                 optimizer=optimizer,
                                 loss='categorical_crossentropy',
                                 metric='accuracy',
                                 name='target')
        return net

    def _calc_y_est(self,
                    x,
                    num_classes,
                    act_type,
                    mdl_conv1a_nf,
                    mdl_conv1b_nf,
                    mdl_conv2a_nf,
                    mdl_conv2b_nf,
                    mdl_fc1_nh,
                    mdl_drop2a_p,
                    mdl_drop2b_p,
                    mdl_drop3_p):
        conv1a = tflearn.conv_2d(x, nb_filter=int(mdl_conv1a_nf), filter_size=3, activation=act_type, name='conv1a')
        conv1b = tflearn.conv_2d(conv1a, nb_filter=int(mdl_conv1b_nf), filter_size=3, activation=act_type, name='conv1b')
        pool1 = tflearn.max_pool_2d(conv1b, kernel_size=2, name='pool1')

        conv2a = tflearn.conv_2d(pool1, nb_filter=int(mdl_conv2a_nf), filter_size=3, activation=act_type, name='conv2a')
        drop2a = tflearn.dropout(conv2a, keep_prob=(1.0 - mdl_drop2a_p), name='drop2b')
        conv2b = tflearn.conv_2d(drop2a, nb_filter=int(mdl_conv2b_nf), filter_size=3, activation=act_type, name='conv2b')
        drop2b = tflearn.dropout(conv2b, keep_prob=(1.0 - mdl_drop2b_p), name='drop2b')
        pool2 = tflearn.max_pool_2d(drop2b, kernel_size=2, name='pool2')

        flatten = tflearn.flatten(pool2, name='flatten')
        fc1 = tflearn.fully_connected(flatten, n_units=int(mdl_fc1_nh), activation=act_type, name='fc1')
        drop3 = tflearn.dropout(fc1, keep_prob=(1.0 - mdl_drop3_p), name='drop3')
        softmax = tflearn.fully_connected(drop3, n_units=num_classes, activation='softmax', name='fc2')
        return softmax
