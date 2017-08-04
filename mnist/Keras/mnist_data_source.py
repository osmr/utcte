import numpy as np

from mnist.common.mnist_data import MnistData
from mnist.common.mnist_data_source_template import MnistDataSourceTemplate
#from mnist_iter import MnistIter


class MnistDataSource(MnistDataSourceTemplate):

    def __init__(self,
                 use_augmentation=True,
                 data_h5_path=None):
        super(MnistDataSource, self).__init__(use_augmentation, data_h5_path)
        #self.train_iter = None
        #self.val_iter = None

    def __call__(self, shuffle=True, dat_batch_size=64, **kwargs):
        self._load_data()
        #self._create_iterators(shuffle, dat_batch_size, **kwargs)
        self.batch_size = dat_batch_size
        return self.train_img, self.train_lbl, self.val_img, self.val_lbl

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
        self.train_lbl = self._to_one_hot(self.train_lbl)
        self.val_lbl = self._to_one_hot(self.val_lbl)
        self.data_loaded = True

    def _to4d(self, img):
        return img.reshape(img.shape[0], self.img_h, self.img_w, 1).astype(np.float32) / 255.0

    def _to_one_hot(self, labels_dense):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * self.n_dim
        labels_one_hot = np.zeros((num_labels, self.n_dim))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    #def _create_iterators(self, shuffle, batch_size, **kwargs):

    #    if (self.val_iter is None) or (self.batch_size != batch_size):
    #        self.val_iter = mx.io.PrefetchingIter(mx.io.NDArrayIter(self.val_img, self.val_lbl, int(batch_size)))
    #        if self.use_augmentation:
    #            self.train_iter = mx.io.PrefetchingIter(MnistIter(
    #                data=mx.io.NDArrayIter(self.train_img, self.train_lbl, int(batch_size), shuffle=shuffle),
    #                gaussian_blur_sigma_max=kwargs['dat_gaussian_blur_sigma_max'],
    #                gaussian_noise_sigma_max=kwargs['dat_gaussian_noise_sigma_max'],
    #                perspective_transform_max_pt_deviation=int(kwargs['dat_perspective_transform_max_pt_deviation']),
    #                max_scale_add=kwargs['dat_max_scale_add'],
    #                max_translate=kwargs['dat_max_translate'],
    #                rotate_max_angle_rad=kwargs['dat_rotate_max_angle_rad']))
    #        else:
    #            self.train_iter = mx.io.PrefetchingIter(mx.io.NDArrayIter(self.train_img, self.train_lbl, int(batch_size), shuffle=shuffle))
    #    else:
    #        self.val_iter.reset()
    #        self.train_iter.reset()
    #        if self.use_augmentation:
    #            mnist_iter = self.train_iter.iters[0]
    #            mnist_iter.gaussian_blur_sigma_max=kwargs['dat_gaussian_blur_sigma_max']
    #            mnist_iter.gaussian_noise_sigma_max=kwargs['dat_gaussian_noise_sigma_max']
    #            mnist_iter.perspective_transform_max_pt_deviation=int(kwargs['dat_perspective_transform_max_pt_deviation'])
    #            mnist_iter.max_scale_add=kwargs['dat_max_scale_add']
    #            mnist_iter.max_translate=kwargs['dat_max_translate']
    #            mnist_iter.rotate_max_angle_rad=kwargs['dat_rotate_max_angle_rad']

    #    self.batch_size = batch_size
