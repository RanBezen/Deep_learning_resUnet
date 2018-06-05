# -*- coding:utf-8 -*-
"""
Generator and Discriminator network.
"""
import tensorflow as tf
from tensorflow import concat, variable_scope
from functools import partial


xavier_init = tf.contrib.layers.xavier_initializer
conv2d = partial(tf.layers.conv2d,
                 activation=tf.nn.relu,
                 kernel_initializer=xavier_init(),
                 padding="VALID")
conv2d_linear = partial(tf.layers.conv2d,
                        kernel_initializer=xavier_init(),
                        padding="SAME")
Cropping2D = tf.keras.layers.Cropping2D
UpSampling2D = tf.keras.layers.UpSampling2D
max_pooling2d = tf.layers.max_pooling2d
dropout = tf.layers.dropout


def downsample(in_data, filters, cropping=None, last=False,
               name='DownSample', reuse=False):
    """
    Down-Sample residual of U-Net.
    :param in_data: The data to down-sample.
    :param filters: The filter size for each convolution.
    :param cropping: The cropping tuple to be applied.
    :param last: For the last down-sample - instead of
                   cropping and pooling, do dropout.
    :param name: The name for the down-sample residual.
    :param reuse: Reuse weights or not.
    :return: The residual output.
    """
    with variable_scope(name, reuse=reuse):
        conv = conv2d(in_data, filters, 3)
        conv = conv2d(conv, filters, 3)
        if last:
            return dropout(conv)
        crop = Cropping2D(cropping=cropping)(conv)
        return crop, max_pooling2d(conv, 2, 2)


def upsample(in_data, crop, filters, name='UpSample', reuse=False):
    """
    Up-Sample residual of U-Net.
    :param crop: The cropping to connect
    :param in_data: The data to up-sample.
    :param filters: The filter size for each convolution.
    :param last: For the last down-sample - instead of
                   cropping and pooling, do dropout.
    :param name: The name for the down-sample residual.
    :param reuse: Reuse weights or not.
    :return: The residual output.
    """
    with variable_scope(name, reuse=reuse):
        up = UpSampling2D(size=(2, 2))(in_data)
        conv1 = conv2d(up, filters, 2, padding="SAME")
        merge6 = concat([crop, conv1], axis=3)
        conv2 = conv2d(merge6, filters, 3)
        conv3 = conv2d(conv2, filters, 3)
        return conv3


def unet(in_data, name='UNet', reuse=False):
    """
    Define Unet, you can refer to:
     - http://blog.csdn.net/u014722627/article/details/60883185
     - https://github.com/zhixuhao/unet
    :param in_data: Input data.
    :param name: Name for the unet residual
    :param reuse: Reuse weights or not.
    :return: The result of the last layer of U-Net.
    """
    assert in_data is not None

    with variable_scope(name, reuse=reuse):
        in_data = tf.image.resize_images(in_data, [168, 168])
        # in_data: (10, 128, 128, 1)
        crop1, pool1 = downsample(in_data, 64, 16, name='DownSample1', reuse=reuse)
        crop2, pool2 = downsample(pool1, 128, 4, name='DownSample2', reuse=reuse)
        drop = downsample(pool2, 256, name='DownSample3', reuse=reuse, last=True)
        up1 = upsample(drop, crop2, 128, name='UpSample1', reuse=reuse)
        up2 = upsample(up1, crop1, 64, name='UpSample2', reuse=reuse)
        conv = conv2d(up2, 3, 3, padding="SAME", name='Conv')
        out = conv2d_linear(conv, 3, 1, name='Out')
    return out

"""
Original Settings:
in_data = tf.image.resize_images(in_data, [640, 640])
crop1, pool1 = downsample(in_data, 64, ((90, 90), (90, 90)), name='DownSample1', reuse=reuse)
crop2, pool2 = downsample(pool1,  128, ((41, 41), (41, 41)), name='DownSample2', reuse=reuse)
crop3, pool3 = downsample(pool2,  256, ((16, 17), (16, 17)), name='DownSample3', reuse=reuse)
crop4, pool4 = downsample(pool3, 512,  ((4, 4), (4, 4)),     name='DownSample4', reuse=reuse)
drop = downsample(pool4, 1024, ((4, 4), (4, 4)),            name='DownSample5', reuse=reuse, last=True)
up1 = upsample(drop, crop4, 512, name='UpSample1', reuse=reuse)
up2 = upsample(up1, crop3, 256, name='UpSample2', reuse=reuse)
up3 = upsample(up2, crop2, 128, name='UpSample3', reuse=reuse)
up4 = upsample(up3, crop1, 64, name='UpSample4', reuse=reuse)
conv = conv2d(up4, 3, 3, padding="SAME", name='Conv')
out = conv2d_linear(conv, 1, 1, name='Out')
"""

"""
crop1, pool1 = downsample(in_data, 64, ((20, 20), (20, 20)), name='DownSample1', reuse=reuse)
crop2, pool2 = downsample(pool1,  128, ((10, 10), (10, 10)), name='DownSample2', reuse=reuse)
drop = downsample(pool2,  256, ((3, 3), (3, 3)), name='DownSample3', reuse=reuse, last=True)
up1 = upsample(drop, crop2, 128, name='UpSample1', reuse=reuse)
up2 = upsample(up1, crop1, 64, name='UpSample2', reuse=reuse)
conv = conv2d(up2, 3, 3, padding="SAME", name='Conv')
out = conv2d_linear(conv, 1, 1, name='Out')
"""