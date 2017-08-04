import mxnet as mx
import numpy as np

import math
import random
import scipy.ndimage
from skimage import transform as tf


class MnistIter(mx.io.DataIter):

    def __init__(
            self,
            data,
            gaussian_blur_sigma_max=1.0,
            #intensivity_max_deviations=[0.00, 0.05, 0.01, 0.00125],
            gaussian_noise_sigma_max=0.05,
            perspective_transform_max_pt_deviation=1,
            max_scale_add=1.0/(28.0/2),
            max_translate=2.0,
            rotate_max_angle_rad=math.pi/12):
        super(MnistIter, self).__init__()
        self.data = data
        self.tol = 1e-5
        self.gaussian_blur_sigma_max = gaussian_blur_sigma_max
        #self.intensivity_max_deviations = intensivity_max_deviations
        self.gaussian_noise_sigma_max = gaussian_noise_sigma_max
        self.perspective_transform_max_pt_deviation = perspective_transform_max_pt_deviation
        self.max_scale_add = max_scale_add
        self.max_translate = max_translate
        self.rotate_max_angle_rad = rotate_max_angle_rad

    @property
    def provide_data(self):
        return self.data.provide_data

    @property
    def provide_label(self):
        return self.data.provide_label

    def reset(self):
        self.data.reset()

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel(), pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def iter_next(self):
        return self.data.iter_next()

    def getdata(self):
        #import matplotlib.pyplot as plt
        batch_imgs_list = self.data.getdata()
        #assert len(batch_imgs_list) == 1
        imgs = batch_imgs_list[0].asnumpy()
        for i in range(len(imgs)):
            x = imgs[i,0,:,:]
            #plt.figure(); plt.imshow(x, cmap=plt.cm.gray)
            x = self._perspective_transform(x, self.perspective_transform_max_pt_deviation, self.max_scale_add, self.max_translate, self.rotate_max_angle_rad)
            x = self._gaussian_blur(x, self.gaussian_blur_sigma_max)
            #x = self._add_change_intensivity(x, self.intensivity_max_deviations)
            x = self._gaussian_noise(x, self.gaussian_noise_sigma_max)
            #plt.figure(); plt.imshow(x, cmap=plt.cm.gray)
            imgs[i,0,:,:] = x
        batch_imgs_list_ = [mx.nd.array(imgs)]
        return batch_imgs_list_

    def getlabel(self):
        return self.data.getlabel()

    # def getindex(self):
    #     return self.data.getindex()

    def getpad(self):
        return self.data.getpad()

    def _gaussian_blur(self, x, sigma_max):
        if (sigma_max is not None) and (sigma_max >= self.tol):
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

    # def _add_change_intensivity(self, x, max_deviations):
    #     if max_deviations is not None:
    #         md = np.array(max_deviations)
    #         k = np.random.uniform(-md, md)
    #         x = k[0] + ((1.0 + k[1]) + (k[2] + k[3] * x) * x) * x
    #     return x

    def _gaussian_noise(self, x, sigma_max):
        if (sigma_max is not None) and (sigma_max >= self.tol):
            sigma = random.uniform(0.0, sigma_max)
            x += np.random.randn(*x.shape) * sigma
        return x

    def _perspective_transform(self, x, max_pt_deviation, max_scale_add, max_translate, rotate_max_angle_rad):
        if max_pt_deviation is not None:
            src_pts = np.array([
                (0, 0),
                (x.shape[0]-1, 0),
                (0, x.shape[1]-1),
                (x.shape[0]-1, x.shape[1]-1)], dtype=np.float32)
            dst_pts = src_pts + np.random.uniform(-max_pt_deviation, max_pt_deviation, src_pts.shape)

            if rotate_max_angle_rad is not None:
                a = np.random.uniform(-rotate_max_angle_rad, rotate_max_angle_rad)
                m1 = np.array([0.5*x.shape[0], 0.5*x.shape[1]], dtype=np.float32)
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
            x[:,:] = tf.warp(x[:,:], tform, order=4, mode="reflect").astype(np.float32)
            x = x * scale + x_min
        return x
