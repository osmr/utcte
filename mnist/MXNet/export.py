if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import os
import argparse

from utct.MXNet.converter import Converter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export MXNet model parameters to h5 file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--checkpoint-dir',
        dest='checkpoint_dir',
        help='Destination directory with checkpoint files',
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
        default=1,
        type=int)
    parser.add_argument(
        '--output',
        dest='dst_filepath',
        help='Output file for MXNet model parameters',
        required=True,
        type=str)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    Converter.export_to_h5(
        checkpoint_path=os.path.join(args.checkpoint_dir, args.prefix),
        checkpoint_epoch=args.epoch,
        dst_filepath=args.dst_filepath)


if __name__ == '__main__':
    main()
