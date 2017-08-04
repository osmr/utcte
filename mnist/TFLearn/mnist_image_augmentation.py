import math
import random
import numpy as np
import scipy.ndimage
from skimage import transform as tf
from tflearn import DataAugmentation


class MnistImageAugmentation(DataAugmentation):

    def __init__(self):
        super(MnistImageAugmentation, self).__init__()

    def add_transform(
            self,
            gaussian_blur_sigma_max=1.0,
            #intensivity_max_deviations=[0.05, 0.05, 0.01, 0.00125],
            gaussian_noise_sigma_max=0.05,
            perspective_transform_max_pt_deviation=1,
            max_scale_add=1.0/(28.0/2),
            max_translate=2.0,
            rotate_max_angle_rad=0.2617994): # Pi/12
        self.methods.append(self._transform)
        self.args.append([gaussian_blur_sigma_max,
                          #intensivity_max_deviations,
                          gaussian_noise_sigma_max,
                          perspective_transform_max_pt_deviation,
                          max_scale_add,
                          max_translate,
                          rotate_max_angle_rad])

    def _transform(
            self,
            batch,
            gaussian_blur_sigma_max=None,
            #intensivity_max_deviations=None,
            gaussian_noise_sigma_max=None,
            perspective_transform_max_pt_deviation=None,
            max_scale_add=None,
            max_translate=None,
            rotate_max_angle_rad=None):
        for i in range(len(batch)):
            batch[i] = self._random_flip_leftright(batch[i])
            batch[i] = self._perspective_transform(batch[i], perspective_transform_max_pt_deviation, max_scale_add, max_translate, rotate_max_angle_rad)
            batch[i] = self._gaussian_blur(batch[i], gaussian_blur_sigma_max)
            #batch[i] = self._add_change_intensivity2(batch[i], intensivity_max_deviations)
            batch[i] = self._gaussian_noise(batch[i], gaussian_noise_sigma_max)
        return batch

    def _random_flip_leftright(self, x):
        if bool(random.getrandbits(1)):
            x = np.fliplr(x)
        return x

    def _gaussian_blur(self, x, sigma_max):
        if sigma_max is not None:
            sigma = random.uniform(0.0, sigma_max)
            x_blurred = scipy.ndimage.filters.gaussian_filter(x, sigma)
            if bool(random.getrandbits(1)):
                x = x_blurred
            else:
                x_blurred2 = scipy.ndimage.gaussian_filter(x_blurred, sigma / 1.2)
                max_alpha = 20.0
                alpha = random.uniform(0.0, max_alpha)
                x = x_blurred + alpha * (x_blurred - x_blurred2)
        return x

    # def _add_change_intensivity2(self, x, max_deviations):
    #     if max_deviations is not None:
    #         md = np.array(max_deviations)
    #         k = np.random.uniform(-md, md)
    #         x = k[0] + ((1.0 + k[1]) + (k[2] + k[3] * x) * x) * x
    #     return x

    def _gaussian_noise(self, x, sigma_max):
        if sigma_max is not None:
            sigma = random.uniform(0.0, sigma_max)
            x += np.random.randn(*x.shape) * sigma
        return x

    def _perspective_transform(self, x, max_pt_deviation, max_scale_add, max_translate, rotate_max_angle_rad):
        if max_pt_deviation is not None:
            src_pts = np.array([
                (0, 0),
                (x.shape[0], 0),
                (0, x.shape[1]),
                (x.shape[0], x.shape[1])], dtype=np.float32)
            dst_pts = src_pts + np.random.uniform(-max_pt_deviation, max_pt_deviation, src_pts.shape)

            if rotate_max_angle_rad is not None:
                a = np.random.uniform(-rotate_max_angle_rad, rotate_max_angle_rad)
                m1 = np.array([x.shape[0]/2, x.shape[1]/2], dtype=np.float32)
                m2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
                dst_pts = np.dot((dst_pts - m1), m2) + m1

            if max_scale_add is not None:
                scale = 1.0 + np.random.uniform(-max_scale_add, max_scale_add)
                m = np.array([[scale, 0], [0, scale]], dtype=np.float32)
                dst_pts = np.dot(dst_pts , m)

            if max_translate is not None:
                dst_pts = dst_pts + np.random.uniform(-max_translate, max_translate, 2)

            tform = tf.ProjectiveTransform()
            tform.estimate(src_pts, dst_pts)
            x_max, x_min = np.max(x), np.min(x)
            scale = x_max - x_min
            x = (x - x_min) / scale
            x[:,:,0] = tf.warp(x[:,:,0], tform, order=4, mode="reflect").astype(np.float32)
            x = x * scale + x_min
        return x

