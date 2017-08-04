import lasagne

from common.optimizer_template import OptimizerTemplate


class Optimizer(OptimizerTemplate):

    def __call__(self,
                 loss_or_grads,
                 params,
                 opt_learning_rate=0.001,
                 opt_epsilon=1e-8,
                 **kwargs):
        return lasagne.updates.adam(
            loss_or_grads,
            params,
            learning_rate=opt_learning_rate,
            epsilon=opt_epsilon)
