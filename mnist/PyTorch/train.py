if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from utct.PyTorch.trainer import Trainer
from common.PyTorch.optimizer import Optimizer
from common.train_config import TrainConfig
from common.train_saver import TrainSaver
from mnist_data_source import MnistDataSource
from mnist_model import MnistModel


def main():

    model = MnistModel()
    optimizer = Optimizer()
    data_source = MnistDataSource()

    cfg = TrainConfig()
    cfg.load(
        model,
        optimizer,
        data_source,
        task_name="mnist",
        framework_name='PyTorch')

    saver = TrainSaver(
        cfg.prm['work_dir'],
        cfg.prm['project_name'],
        cfg.prm['model_filename_prefix'],
        data_source,
        task_name="mnist",
        suffix="_pt")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_source=data_source,
        saver=saver)

    trainer.train(
        num_epoch=cfg.prm['max_num_epoch'],
        epoch_tail=cfg.prm['min_num_epoch'])


if __name__ == '__main__':
    main()
