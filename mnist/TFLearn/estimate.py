if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import os
import argparse

from mnist_data_source import MnistDataSource
from mnist_model import MnistModel
from common.TFLearn.optimizer import Optimizer
from utct.TFLearn.estimator import Estimator


def parse_args(framework_name):
    parser = argparse.ArgumentParser(
        description='Estimator of a trained neural net classifier for the MNIST task on the base of {} framework'.format(
            framework_name),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cache-dir',
        dest='data_cache_dir',
        help='Path to data cache directory',
        required=True,
        type=str)
    parser.add_argument(
        '--checkpoint-dir',
        dest='checkpoint_dir',
        help='Path to checkpoint files',
        required=True,
        type=str)
    parser.add_argument(
        '--file',
        dest='file_name',
        help='File name of checkpoint file',
        required=True,
        type=str)
    args = parser.parse_args()
    return args


def main():

    args = parse_args(framework_name='TFLearn')
    model = MnistModel()
    optimizer = Optimizer()
    data_source = MnistDataSource(use_augmentation=False)
    data_source.update_project_dirname(args.data_cache_dir)

    Estimator.estimate(
        model=model,
        optimizer=optimizer,
        data_source=data_source,
        checkpoint_path=os.path.join(args.checkpoint_dir, args.file_name))


if __name__ == '__main__':
    main()
