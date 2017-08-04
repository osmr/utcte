import tensorflow as tf
import tensorflow.contrib.slim as slim

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
                 #optimizer,
                 #data_augmentation,
                 num_classes=10,
                 act_fn=tf.nn.relu,
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

        x_name = "InputData"
        with tf.name_scope(x_name):
            x = tf.placeholder(dtype=tf.float32,
                               shape=(None, self.img_h, self.img_w, 1),
                               name='XX')
        tf.GraphKeys.INPUTS = 'inputs'
        tf.add_to_collection(tf.GraphKeys.INPUTS, x)
        tf.GraphKeys.LAYER_TENSOR = 'layer_tensor'
        tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + x_name, x)

        y = tf.placeholder(dtype=tf.float32,
                           shape=(None, num_classes),
                           name='YY')
        tf.GraphKeys.TARGETS = 'targets'
        tf.add_to_collection(tf.GraphKeys.TARGETS, y)

        if is_training:
            global_is_training = tf.get_collection('is_training')[0]
            if global_is_training:
                is_training = False

        y_est = self._calc_y_est(x,
                                 num_classes,
                                 act_fn,
                                 mdl_conv1a_nf,
                                 mdl_conv1b_nf,
                                 mdl_conv2a_nf,
                                 mdl_conv2b_nf,
                                 mdl_fc1_nh,
                                 mdl_drop2a_p if is_training else 0.0,
                                 mdl_drop2b_p if is_training else 0.0,
                                 mdl_drop3_p if is_training else 0.0,
                                 is_training)
        #loss = slim.losses.softmax_cross_entropy(y_est, y)
        loss = self._categorical_crossentropy(y_est, y)
        metric = self._accuracy(y_est, y)
        return y_est, loss, metric

    def _calc_y_est(self,
                    x,
                    num_classes,
                    act_fn,
                    mdl_conv1a_nf,
                    mdl_conv1b_nf,
                    mdl_conv2a_nf,
                    mdl_conv2b_nf,
                    mdl_fc1_nh,
                    mdl_drop2a_p,
                    mdl_drop2b_p,
                    mdl_drop3_p,
                    is_training):
        conv1a = slim.conv2d(inputs=x, num_outputs=int(mdl_conv1a_nf), kernel_size=3, activation_fn=act_fn, scope='conv1a')
        conv1b = slim.conv2d(inputs=conv1a, num_outputs=int(mdl_conv1b_nf), kernel_size=3, activation_fn=act_fn, scope='conv1b')
        pool1 = slim.max_pool2d(inputs=conv1b, kernel_size=2, scope='pool1')

        conv2a = slim.conv2d(inputs=pool1, num_outputs=int(mdl_conv2a_nf), kernel_size=3, activation_fn=act_fn, scope='conv2a')
        drop2a = slim.dropout(inputs=conv2a, keep_prob=(1.0 - mdl_drop2a_p), is_training=is_training, scope='drop2a')
        conv2b = slim.conv2d(inputs=drop2a, num_outputs=int(mdl_conv2b_nf), kernel_size=3, activation_fn=act_fn, scope='conv2b')
        drop2b = slim.dropout(inputs=conv2b, keep_prob=(1.0 - mdl_drop2b_p), is_training=is_training, scope='drop2b')
        pool2 = slim.max_pool2d(inputs=drop2b, kernel_size=2, scope='pool2')

        flatten = slim.flatten(inputs=pool2, scope='flatten')
        fc1 = slim.fully_connected(inputs=flatten, num_outputs=int(mdl_fc1_nh), activation_fn=act_fn, scope='fc1')
        drop3 = slim.dropout(inputs=fc1, keep_prob=(1.0 - mdl_drop3_p), is_training=is_training, scope='drop3')
        softmax = slim.fully_connected(inputs=drop3, num_outputs=num_classes, activation_fn=tf.nn.softmax, scope='fc2')
        return softmax

    def _categorical_crossentropy(self, y_pred, y_true):
        EPSILON = 1e-10
        with tf.name_scope("Crossentropy"):
            y_pred /= tf.reduce_sum(y_pred,
                                    reduction_indices=len(y_pred.get_shape())-1,
                                    keep_dims=True)
            # manual computation of crossentropy
            y_pred = tf.clip_by_value(y_pred,
                                      tf.cast(EPSILON, dtype=tf.float32),
                                      tf.cast(1.-EPSILON, dtype=tf.float32))
            cross_entropy = - tf.reduce_sum(
                y_true * tf.log(y_pred),
                reduction_indices=len(y_pred.get_shape())-1)
            return tf.reduce_mean(cross_entropy)

    def _accuracy(self, y_pred, y_true):
        with tf.name_scope('Accuracy'):
            correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="name")
        return acc
