if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import os
from sys import platform
import argparse

import mxnet as mx
from mnist_model import MnistModel
from mnist_data_source import MnistDataSource
from utct.MXNet.converter import Converter


def parse_args():
    is_win_os = "win" in platform.lower()
    parser = argparse.ArgumentParser(
        description='Import MXNet model parameters from h5 file',
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
        '--input',
        dest='src_filepath',
        help='Input file with MXNet model parameters',
        required=True,
        type=str)
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

    args = parse_args()
    model = MnistModel()
    data_source = MnistDataSource(use_augmentation=False)
    data_source.update_project_dirname(args.data_cache_dir)
    ctx = [mx.gpu(i) for i in args.gpus] if args.gpus else mx.cpu()
    Converter.import_from_h5(
        model=model,
        data_source=data_source,
        src_filepath=args.src_filepath,
        checkpoint_path=os.path.join(args.checkpoint_dir, args.prefix),
        checkpoint_epoch=args.epoch,
        ctx=ctx)


if __name__ == '__main__':
    main()
