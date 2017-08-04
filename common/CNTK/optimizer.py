import cntk

from common.optimizer_template import OptimizerTemplate


class Optimizer(OptimizerTemplate):

    def __call__(self,
                 parameters,
                 opt_learning_rate=0.001,
                 **kwargs):
        lr_per_minibatch = cntk.learning_rate_schedule(
            lr=opt_learning_rate,
            unit=cntk.UnitType.minibatch)
        momentum = cntk.momentum_schedule(
            momentum=0.99)
        return cntk.adam_sgd(
            parameters=parameters,
            lr=lr_per_minibatch,
            momentum=momentum)
