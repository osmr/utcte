import tensorflow as tf

from common.optimizer_template import OptimizerTemplate


class Optimizer(OptimizerTemplate):

    def __call__(self,
                 opt_learning_rate=0.001,
                 opt_epsilon=1e-8,
                 **kwargs):
        return tf.train.AdamOptimizer(
            learning_rate=opt_learning_rate,
            epsilon=opt_epsilon)