from utct.common.functor import Functor


class OptimizerTemplate(Functor):

    def __init__(self):
        super(OptimizerTemplate, self).__init__()
        self.param_bounds = {
           'opt_learning_rate': (1e-6, 1e-3),
           'opt_epsilon': (1e-8, 1e-3)}