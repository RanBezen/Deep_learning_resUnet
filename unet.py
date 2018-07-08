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


def residualDownsample(in_data, filters, cropping=None, last=False,
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
        in_data=batch_norm(in_data,name='batchNormalization')        
        conv = conv2d(in_data, filters, 3)
        conv = conv2d(conv, filters, 3)
        identity=Cropping2D(cropping=2)(in_data)
        identityShortcut=concat([identity,conv],axis=3)
        resBlock=residualBlock(identityShortcut,filters,3)
        if last:
          return dropout(resBlock)
        crop = Cropping2D(cropping=cropping)(resBlock)
        pool=max_pooling2d(resBlock, 2, 2)
        return crop, pool

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


def residualUpsample(in_data, crop, filters, name='UpSample', reuse=False):
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
        in_data=batch_norm(in_data,name='batchNormalization')
        up = UpSampling2D(size=(2, 2))(in_data)
        conv1 = conv2d(up, filters, 2, padding="SAME")
        merge6 = concat([crop, conv1], axis=3)
        conv2 = conv2d(merge6, filters, 3)
        conv3 = conv2d(conv2, filters, 3)
        identity=Cropping2D(cropping=2)(merge6)
        identityShortcut=concat([identity,conv3],axis=3)
        resBlock=residualBlock(identityShortcut,filters,3)
        
        return resBlock

def residualBlock(input,channels, kernel,name='residualBlock', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        conv=conv2d_linear(input,channels, kernel)
        conv=conv2d_linear(conv,channels,kernel)
        shortcut=concat([input,conv],axis=3)
        conv=tf.nn.relu(shortcut)
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
        h=in_data.get_shape().as_list()[1]
        w = in_data.get_shape().as_list()[2]
        in_data = tf.image.resize_images(in_data, [h+40, w+40])
        in_data = tf.image.resize_images(in_data, [168, 168])
        # in_data: (10, 128, 128, 1)
        
        crop1, pool1 = residualDownsample(in_data, 64, 16, name='DownSample1', reuse=reuse)
        crop2, pool2 = residualDownsample(pool1, 128, 4, name='DownSample2', reuse=reuse)
        drop = residualDownsample(pool2, 256, name='DownSample3', reuse=reuse, last=True)
        up1 = residualUpsample(drop, crop2, 128, name='UpSample1', reuse=reuse)
        up2 = residualUpsample(up1, crop1, 64, name='UpSample2', reuse=reuse)
        conv = conv2d(up2, 3, 3, padding="SAME", name='Conv')
        out = conv2d_linear(conv, 3, 1, name='Out')
    return out
