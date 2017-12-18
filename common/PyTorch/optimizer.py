import torch

from common.optimizer_template import OptimizerTemplate


class Optimizer(OptimizerTemplate):

    def __call__(self,
                 params,
                 opt_learning_rate=0.001,
                 opt_epsilon=1e-8,
                 **kwargs):
        return torch.optim.Adam(
            params=params,
            lr=opt_learning_rate,
            eps=opt_epsilon)

