if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import os
from sys import platform
import argparse
import mxnet as mx

from mnist_data_source import MnistDataSource
from utct.MXNet.estimator import Estimator


def parse_args(framework_name):
    is_win_os = "win" in platform.lower()
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
        '--prefix',
        dest='prefix',
        help='Prefix for MXNet checkpoint files',
        default='mnist',
        type=str)
    parser.add_argument(
        '--epoch',
        dest='epoch',
        help='Epoch for MXNet checkpoint files',
        required=True,
        type=int)
    parser.add_argument(
        '--gpus',
        dest='gpus',
        help='List of GPU device numbers to train with, empty is CPU',
        default=([] if is_win_os else [0]),
        nargs='*',
        type=int)
    args = parser.parse_args()
    return args


def main():

    args = parse_args(framework_name='MXNet')
    data_source = MnistDataSource(use_augmentation=False)
    data_source.update_project_dirname(args.data_cache_dir)

    ctx = [mx.gpu(i) for i in args.gpus] if args.gpus else mx.cpu()
    Estimator.estimate(
        data_source=data_source,
        checkpoint_path=os.path.join(args.checkpoint_dir, args.prefix),
        checkpoint_epoch=args.epoch,
        ctx=ctx)


if __name__ == '__main__':
    main()
