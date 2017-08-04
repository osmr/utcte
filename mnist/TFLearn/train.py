if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from common.train_config import TrainConfig
from common.train_saver import TrainSaver
from mnist_model import MnistModel
from common.TFLearn.optimizer import Optimizer
from mnist_data_source import MnistDataSource
from utct.TFLearn.trainer import Trainer


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
        framework_name='TFLearn')

    saver = TrainSaver(
        cfg.prm['work_dir'],
        cfg.prm['project_name'],
        cfg.prm['model_filename_prefix'],
        data_source=data_source,
        task_name="mnist",
        suffix="_tfl")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_source=data_source,
        saver=saver)

    # trainer.train(
    #     num_epoch=cfg.prm['max_num_epoch'],
    #     epoch_tail = cfg.prm['min_num_epoch'],
    #     dat_gaussian_blur_sigma_max=1.0,
    #     dat_gaussian_noise_sigma_max=0.05,
    #     dat_perspective_transform_max_pt_deviation=1,
    #     dat_max_scale_add=1.0 / (28.0 / 2),
    #     dat_max_translate=2.0,
    #     dat_rotate_max_angle_rad=0.2617994)

    trainer.hyper_train(num_epoch=cfg.prm['max_num_epoch'],
                        epoch_tail=cfg.prm['min_num_epoch'],
                        bo_num_iter=cfg.prm['bo_num_iter'],
                        bo_kappa=cfg.prm['bo_kappa'],
                        bo_min_rand_num=cfg.prm['bo_min_rand_num'],
                        bo_results_filename='mnist_hyper.csv',
                        synch_file_list=cfg.prm['synch_list'],
                        sync_period=cfg.prm['sync_period'])


if __name__ == '__main__':
    main()
