import os
import math

from utct.common.data_source_template import DataSourceTemplate


class MnistDataSourceTemplate(DataSourceTemplate):

    def __init__(self,
                 use_augmentation=True,
                 data_h5_path=None):
        super(MnistDataSourceTemplate, self).__init__(use_augmentation)
        self.param_bounds = {
            #'dat_batch_size': (10, 200),
            'dat_gaussian_blur_sigma_max': (0.0, 1.0),
            'dat_gaussian_noise_sigma_max': (0.0, 0.05),
            'dat_perspective_transform_max_pt_deviation': (0.0, 2.99),
            'dat_max_scale_add': (0.0, 2.0 / (28.0 / 2)),
            'dat_max_translate': (0.0, 3.0),
            'dat_rotate_max_angle_rad': (0.0, math.pi / 12)}
        #self.param_bounds = {
        #    'dat_batch_size': (10, 200)}
        self.params = {
            'data_h5_path': '../TEMP/mnist/mnist.h5'}
        self.use_augmentation = use_augmentation
        if data_h5_path is not None:
            self.params['data_h5_path'] = data_h5_path

        self.n_dim = 10
        self.img_h = 28
        self.img_w = 28
        self.cache_data_dirname = None

        self.data_loaded = False
        self.train_img = None
        self.train_lbl = None
        self.val_img = None
        self.val_lbl = None
        self.batch_size = None

    def update_project_dirname(self, project_dirname):
        self.cache_data_dirname = os.path.join(project_dirname, 'cache_data')
        if not os.path.exists(self.cache_data_dirname):
            os.makedirs(self.cache_data_dirname)

    def update_cache_data_dirname(self, cache_data_dirname):
        self.cache_data_dirname = cache_data_dirname
