if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import os
import argparse

from mnist_model import MnistModel
from common.TFLearn.optimizer import Optimizer
from utct.TFLearn.converter import Converter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export TFLearn model parameters to h5 file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument(
        '--output',
        dest='dst_filepath',
        help='Output file for TFLearn model parameters',
        required=True,
        type=str)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    model = MnistModel()
    optimizer = Optimizer()

    Converter.export_to_h5(
        model=model,
        optimizer=optimizer,
        checkpoint_path=os.path.join(args.checkpoint_dir, args.file_name),
        dst_filepath=args.dst_filepath)


if __name__ == '__main__':
    main()
