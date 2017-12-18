from mxnet import gluon

from common.optimizer_template import OptimizerTemplate


class Optimizer(OptimizerTemplate):

    def __call__(self,
                 params,
                 opt_learning_rate=0.001,
                 opt_epsilon=1e-8,
                 **kwargs):
        return gluon.Trainer(
            params=params,
            optimizer='adam',
            optimizer_params={'learning_rate': opt_learning_rate, 'epsilon': opt_epsilon})

