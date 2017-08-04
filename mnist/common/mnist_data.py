import sys
import gzip
import os
import struct
import urllib
import shutil

import h5py
import numpy as np


class MnistData(object):
    """
    Data provider class for MNIST dataset.

    Parameters:
    ----------
    work_dirname : str
        directory name for storing temp and cache files
    cached_filename : str
        file name for cache of data
    """

    def __init__(self,
                 work_dirname="",
                 cached_filename= 'mnist.h5',
                 data_h5_path=None):

        self.work_dirname = work_dirname
        self.checked = False
        self.data_filename = os.path.join(work_dirname, cached_filename)
        self.data_h5_path = data_h5_path

    def check_cache(self):
        """
        Check cached data. If data doesn't exist, we will create it.
        """

        if os.path.exists(self.data_filename):
            self.checked = True
            return

        if not os.path.exists(self.work_dirname):
            os.makedirs(self.work_dirname)

        if (self.data_h5_path is not None) and self.data_h5_path and os.path.exists(self.data_h5_path):
            shutil.copy(self.data_h5_path, self.data_filename)
            self.checked = True
            return

        src_url_prefix = 'http://yann.lecun.com/exdb/mnist/'
        train_lbl_filename = 'train-labels-idx1-ubyte.gz'
        train_img_filename = 'train-images-idx3-ubyte.gz'
        val_lbl_filename = 't10k-labels-idx1-ubyte.gz'
        val_img_filename = 't10k-images-idx3-ubyte.gz'

        train_lbl, train_img = self._read_data(
            src_url_prefix, self.work_dirname, train_lbl_filename, train_img_filename)
        val_lbl, val_img = self._read_data(
            src_url_prefix, self.work_dirname, val_lbl_filename, val_img_filename)

        h5f = h5py.File(self.data_filename, 'w')
        h5f.create_dataset('train_lbl', data=train_lbl)
        h5f.create_dataset('train_img', data=train_img)
        h5f.create_dataset('val_lbl', data=val_lbl)
        h5f.create_dataset('val_img', data=val_img)
        h5f.close()

        self.checked = True

    def get_data(self):
        """
        Create numpy arrays for training and validation data sets.

        Returns:
        ----------
        grayscale images as np-arrays and labels as numbers
        """

        if not self.checked:
            self.check_cache()
        h5f = h5py.File(self.data_filename, 'r')
        train_lbl = h5f['train_lbl'][:]
        train_img = h5f['train_img'][:]
        val_lbl = h5f['val_lbl'][:]
        val_img = h5f['val_img'][:]
        h5f.close()
        return train_img, train_lbl, val_img, val_lbl

    def _download_data(self, src_url_prefix, dst_dirname, filename, force_download=False):
        filepath = os.path.join(dst_dirname, filename)
        if force_download or not os.path.exists(filepath):
            url = src_url_prefix + filename
            if sys.version_info.major == 2:
                urllib.urlretrieve(url, filepath)
            else:
                urllib.request.urlretrieve(url, filepath)
        return filepath

    def _read_data(
            self,
            src_url_prefix,
            dst_dirname,
            label_filename,
            image_filename,
            clear_files=True):
        label_path = self._download_data(
            src_url_prefix, dst_dirname, label_filename)
        with gzip.open(label_path) as flbl:
            _, _ = struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.int8)
        image_path = self._download_data(
            src_url_prefix, dst_dirname, image_filename)
        with gzip.open(image_path, 'rb') as fimg:
            _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
                len(label), rows, cols)
        if clear_files:
            os.remove(label_path)
            os.remove(image_path)
        return label, image
