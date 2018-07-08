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
batch_norm = partial(tf.layers.batch_normalization, training=True)





def residualDownsample(in_data, filters,  last=False,
                       name='DownSample', reuse=False):

    with variable_scope(name, reuse=reuse):
        in_data = batch_norm(in_data, name='batchNormalization')
        identityShortcut = residualBlock(in_data,filters, 3)
        if last:
            return dropout(identityShortcut)
        pool = max_pooling2d(identityShortcut, 2, 2)
        return identityShortcut, pool



def residualUpsample(in_data, crop, filters, name='UpSample', reuse=False):

    with variable_scope(name, reuse=reuse):
        in_data = batch_norm(in_data, name='batchNormalization')
        up = UpSampling2D(size=(2, 2))(in_data)
        merge6 = concat([crop, up], axis=3)
        identityShortcut = residualBlock(merge6, filters,3)

        return identityShortcut


def residualBlock(input, channels, kernel, name='residualBlock', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        conv = conv2d_linear(input, channels, kernel)
        conv = conv2d_linear(conv, channels, kernel)
        shortcut = input+ conv
        conv = tf.nn.relu(shortcut)
        conv = conv2d_linear(conv, channels, kernel)
        conv = conv2d_linear(conv, channels, kernel)
        shortcut = input + conv
        conv = tf.nn.relu(shortcut)
    return conv


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
        # in_data: (10, 128, 128, 1)

        crop1, pool1 = residualDownsample(in_data, 64, name='DownSample1', reuse=reuse)
        crop2, pool2 = residualDownsample(pool1, 128,name='DownSample2', reuse=reuse)
        drop = residualDownsample(pool2, 256, name='DownSample3', reuse=reuse, last=True)
        up1 = residualUpsample(drop, crop2, 128, name='UpSample1', reuse=reuse)
        up2 = residualUpsample(up1, crop1, 64, name='UpSample2', reuse=reuse)
        conv = conv2d(up2, 3, 3, padding="SAME", name='Conv')
        out = conv2d_linear(conv, 3, 1, name='Out')
    return out
