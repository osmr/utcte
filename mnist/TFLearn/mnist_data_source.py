import numpy as np

from mnist.common.mnist_data import MnistData
from mnist.common.mnist_data_source_template import MnistDataSourceTemplate
from utct.TFLearn.tflearn_data_source_extra_template import TflearnDataSourceExtraTemplate
from mnist_image_augmentation import MnistImageAugmentation


class MnistDataSource(MnistDataSourceTemplate, TflearnDataSourceExtraTemplate):

    def __init__(self,
                 use_augmentation=True,
                 rewrite_data_aug=False,
                 data_h5_path=None):
        MnistDataSourceTemplate.__init__(self, use_augmentation, data_h5_path)
        TflearnDataSourceExtraTemplate.__init__(self, rewrite_data_aug)
        self.img_aug = None

    def __call__(self, batch_size=64, **kwargs):
        self._load_data()
        self.batch_size = batch_size
        self._update_augmentator(**kwargs)
        return {
            "X_inputs": self.train_img,
            "Y_targets": self.train_lbl,
            "validation_set": (self.val_img, self.val_lbl)}

    def _update_augmentator(self, **kwargs):
        if self.use_augmentation:
            self.img_aug = MnistImageAugmentation()
            self.img_aug.add_transform(
                gaussian_blur_sigma_max=kwargs['dat_gaussian_blur_sigma_max'],
                gaussian_noise_sigma_max=kwargs['dat_gaussian_noise_sigma_max'],
                perspective_transform_max_pt_deviation=int(kwargs['dat_perspective_transform_max_pt_deviation']),
                max_scale_add=kwargs['dat_max_scale_add'],
                max_translate=kwargs['dat_max_translate'],
                rotate_max_angle_rad=kwargs['dat_rotate_max_angle_rad'])

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
