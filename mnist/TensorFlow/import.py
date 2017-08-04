if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import os
import argparse

from mnist_model import MnistModel
from utct.TensorFlow.converter import Converter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Import TensorFlow model parameters from h5 file',
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
        '--input',
        dest='src_filepath',
        help='Input file with TensorFlow model parameters',
        required=True,
        type=str)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    model = MnistModel()

    Converter.import_from_h5(
        model=model,
        src_filepath=args.src_filepath,
        checkpoint_path=os.path.join(args.checkpoint_dir, args.file_name))


if __name__ == '__main__':
    main()
