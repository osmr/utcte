import numpy as np
from mxnet import gluon
from mnist_dataset import MnistDataset
from mnist.common.mnist_data import MnistData
from mnist.common.mnist_data_source_template import MnistDataSourceTemplate


class MnistDataSource(MnistDataSourceTemplate):

    def __init__(self,
                 use_augmentation=True,
                 data_h5_path=None):
        super(MnistDataSource, self).__init__(use_augmentation, data_h5_path)
        self.train_loader = None
        self.val_loader = None

    def __call__(self,
                 shuffle=True,
                 dat_batch_size=64,
                 **kwargs):
        self._load_data()
        self.batch_size = dat_batch_size

        self.train_loader = gluon.data.DataLoader(
            MnistDataset(self.train_img, self.train_lbl),
            batch_size=dat_batch_size,
            shuffle=True,
            last_batch='discard')

        self.val_loader = gluon.data.DataLoader(
            MnistDataset(self.val_img, self.val_lbl),
            batch_size=dat_batch_size,
            shuffle=False)

        return self.train_loader, self.val_loader

    def _load_data(self):
        if self.data_loaded:
            return
        if self.cache_data_dirname is None:
            raise Exception("Error: Project directory name is unknown")
        self.train_img, self.train_lbl, self.val_img, self.val_lbl = MnistData(
            work_dirname=self.cache_data_dirname,
            data_h5_path=self.params['data_h5_path']).get_data()
        self.train_img = self._to4d(self.train_img)
        self.val_img = self._to4d(self.val_img)
        #self.train_lbl = self.train_lbl.astype(np.long)
        #self.val_lbl = self.val_lbl.astype(np.long)
        self.data_loaded = True

    def _to4d(self, img):
        return img.reshape(img.shape[0], 1, self.img_h, self.img_w).astype(np.float32) / 255.0
