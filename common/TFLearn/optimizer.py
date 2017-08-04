import tflearn

from common.optimizer_template import OptimizerTemplate


class Optimizer(OptimizerTemplate):

    def __call__(self,
                 #opt_learning_rate=0.001,
                 #opt_epsilon=1e-8,
                 **kwargs):
        return tflearn.Adam(
            learning_rate=(kwargs['opt_learning_rate'] if 'opt_learning_rate' in kwargs else 0.001),
            epsilon=(kwargs['opt_epsilon'] if 'opt_epsilon' in kwargs else 1e-8))
