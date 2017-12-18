from mxnet.gluon import nn

from utct.common.functor import Functor


class MnistModel(Functor):

    def __init__(self):
        super(MnistModel, self).__init__()

    def __call__(self):
        net = nn.Sequential()
        with net.name_scope():
            net.add(nn.Dense(128, activation='relu'))
            net.add(nn.Dense(64, activation='relu'))
            net.add(nn.Dense(10))
        return net


