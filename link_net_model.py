# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

'''
LinkNet

# Reference
- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385)
- [LinkNet](https://arxiv.org/pdf/1707.03718.pdf)
- [LinkNet Project](https://codeac29.github.io/projects/linknet/)

# Pretrained model weights
- [Download pretrained resnet-34 model]
  (https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5)

'''


class LinkNet:
    def __init__(self, pretrained_weights, is_training, data_format='channels_first', num_classes=15):
        self._weights_h5 = h5py.File(pretrained_weights, 'r')
        self._is_training = is_training
        self._data_format = data_format
        self._num_classes = num_classes
        self._padding = 'SAME'
        self._feature_map_axis = None
        self._encoder_data_format = None
        self._pool_kernel = None
        self._pool_strides = None
        self._res_conv_strides = [1, 1, 1, 1]
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

        '''
        based on the data format set appropriate pool_kernel and pool_strides
        always use channels_first i.e. NCHW as the data format on a GPU
        '''

        if data_format == 'channels_first':
            self._encoder_data_format = 'NCHW'
            self._pool_kernel = [1, 1, 3, 3]
            self._pool_strides = [1, 1, 2, 2]
            self._feature_map_axis = 1
        else:
            self._encoder_data_format = 'NHWC'
            self._pool_kernel = [1, 3, 3, 1]
            self._pool_strides = [1, 2, 2, 1]
            self._feature_map_axis = -1

    # define resnet-34 encoder
    def resnet34_encoder(self, features):
        # input : BGR format in range [0-255]

        if self._data_format == 'channels_last':
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        # Stage 0
        self.stage0 = self._conv_layer(
            features, 'conv0', strides=self._pool_strides)
        self.stage0 = self._get_batchnorm_layer(self.stage0, 'bn0')
        self.stage0 = self._get_relu_activation(self.stage0, name='relu0')
        # 64

        # Stage 1
        self.stage1 = tf.nn.max_pool(self.stage0, ksize=self._pool_kernel, strides=self._pool_strides,
                                     padding=self._padding, data_format=self._encoder_data_format, name='maxpool1')
        # 64

        # Stage 2
        self.stage2 = self._res_conv_block(
            input_layer=self.stage1, stage='stage1_unit1_', strides=self._res_conv_strides)
        self.stage2 = self._res_identity_block(
            input_layer=self.stage2, stage='stage1_unit2_')
        self.stage2 = self._res_identity_block(
            input_layer=self.stage2, stage='stage1_unit3_')
        # 64

        # Stage 3
        self.stage3 = self._res_conv_block(
            input_layer=self.stage2, stage='stage2_unit1_', strides=self._pool_strides)
        self.stage3 = self._res_identity_block(
            input_layer=self.stage3, stage='stage2_unit2_')
        self.stage3 = self._res_identity_block(
            input_layer=self.stage3, stage='stage2_unit3_')
        self.stage3 = self._res_identity_block(
            input_layer=self.stage3, stage='stage2_unit4_')
        # 128

        # Stage 4
        self.stage4 = self._res_conv_block(
            input_layer=self.stage3, stage='stage3_unit1_', strides=self._pool_strides)
        self.stage4 = self._res_identity_block(
            input_layer=self.stage4, stage='stage3_unit2_')
        self.stage4 = self._res_identity_block(
            input_layer=self.stage4, stage='stage3_unit3_')
        self.stage4 = self._res_identity_block(
            input_layer=self.stage4, stage='stage3_unit4_')
        self.stage4 = self._res_identity_block(
            input_layer=self.stage4, stage='stage3_unit5_')
        self.stage4 = self._res_identity_block(
            input_layer=self.stage4, stage='stage3_unit6_')
        # 256

        # Stage 5
        self.stage5 = self._res_conv_block(
            input_layer=self.stage4, stage='stage4_unit1_', strides=self._pool_strides)
        self.stage5 = self._res_identity_block(
            input_layer=self.stage5, stage='stage4_unit2_')
        self.stage5 = self._res_identity_block(
            input_layer=self.stage5, stage='stage4_unit3_')
        self.stage5 = self._get_batchnorm_layer(self.stage5, 'bn1')
        self.stage5 = self._get_relu_activation(self.stage5, name='relu1')
        # 512

    # define link-net decoder
    def link_net_decoder(self):
        self.decoder1_out = self._get_decoder_block(
            self.stage5, 512, 256, [2, 2], name='decoder1_')
        self.decoder1_fuse = tf.add(
            self.decoder1_out, self.stage4, name='decoder1_fuse')
        self.decoder2_out = self._get_decoder_block(
            self.decoder1_fuse, 256, 128, [2, 2], name='decoder2_')
        self.decoder2_fuse = tf.add(
            self.decoder2_out, self.stage3, name='decoder2_fuse')
        self.decoder3_out = self._get_decoder_block(
            self.decoder2_fuse, 128, 64, [2, 2], name='decoder3_')
        self.decoder3_fuse = tf.add(
            self.decoder3_out, self.stage2, name='decoder3_fuse')
        self.decoder4_out = self._get_decoder_block(
            self.decoder3_fuse, 64, 64, [1, 1], name='decoder4_')

        self.decoder5_conv_tr = self._get_conv2d_transpose_layer(
            self.decoder4_out, 32, [3, 3], [2, 2], name='decoder5_conv_tr')
        self.decoder5_bn_tr = self._get_batchnorm_layer(
            self.decoder5_conv_tr, name='decoder5_bn_tr')
        self.decoder5_relu_tr = self._get_relu_activation(
            self.decoder5_bn_tr, name='decoder5_relu_tr')

        self.decoder5_conv = self._get_conv2d_layer(
            self.decoder5_relu_tr, 32, [3, 3], [1, 1], name='decoder5_conv')
        self.decoder5_bn = self._get_batchnorm_layer(
            self.decoder5_conv, name='decoder5_bn')
        self.decoder5_relu = self._get_relu_activation(
            self.decoder5_bn, name='decoder5_relu')

        self.logits = self._get_conv2d_transpose_layer(
            self.decoder5_relu, self._num_classes, [3, 3], [2, 2], name='logits')

    # build decoder residual block
    def _get_decoder_block(self, input_layer, in_kernels, out_kernels, strides, name='decoder_'):
        x = self._get_conv2d_layer(input_layer, int(
            in_kernels // 4), [1, 1], [1, 1], name=name + 'conv1')
        x = self._get_batchnorm_layer(x, name=name + 'bn1')
        x = self._get_relu_activation(x, name=name + 'relu1')

        x = self._get_conv2d_transpose_layer(
            x, int(in_kernels // 4), [3, 3], strides, name=name + 'conv_tr')
        x = self._get_batchnorm_layer(x, name=name + 'bn_tr')
        x = self._get_relu_activation(x, name=name + 'relu_tr')

        x = self._get_conv2d_layer(x, out_kernels, [1, 1], [
                                   1, 1], name=name + 'conv2')
        x = self._get_batchnorm_layer(x, name=name + 'bn2')
        x = self._get_relu_activation(x, name=name + 'relu2')

        return x

    # return convolution2d layer
    def _get_conv2d_layer(self, input_tensor, num_filters, kernel_size, strides, name='conv'):
        return tf.layers.conv2d(inputs=input_tensor, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)

    # return convolution2d_transpose layer
    def _get_conv2d_transpose_layer(self, input_tensor, num_filters, kernel_size, strides, name='conv_tr'):
        return tf.layers.conv2d_transpose(inputs=input_tensor, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)

    # return relu activation function
    def _get_relu_activation(self, input_tensor, name='relu'):
        return tf.nn.relu(input_tensor, name=name)

    # return the dropout layer
    def _get_dropout_layer(self, input_tensor, rate=0.5, name='dropout'):
        return tf.layers.dropout(inputs=input_tensor, rate=rate, training=self._is_training, name=name)

    # return batch normalization layer
    def _get_batchnorm_layer(self, input_tensor, name='bn'):
        return tf.layers.batch_normalization(input_tensor, axis=self._feature_map_axis, training=self._is_training, name=name)

    #-------------------------------------#
    # pretrained resnet encoder functions #
    #-------------------------------------#
    #-----------------------#
    # convolution layer     #
    #-----------------------#
    def _conv_layer(self, input_layer, name, strides=[1, 1, 1, 1]):
        hierarchy_name = list(self._weights_h5[name])[0]
        W_init_value = np.array(
            self._weights_h5[name][hierarchy_name]['kernel:0'], dtype=np.float32)
        W = tf.get_variable(name=name + '_kernel', shape=W_init_value.shape,
                            initializer=tf.constant_initializer(W_init_value), dtype=tf.float32)
        x = tf.nn.conv2d(input_layer, filter=W, strides=strides, padding=self._padding,
                         data_format=self._encoder_data_format, name=name + 'conv')

        return x

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _res_conv_block(self, input_layer, stage, strides):
        x = self._get_batchnorm_layer(input_layer, name=stage + 'bn1')
        x = self._get_relu_activation(x, name=stage + 'relu1')

        shortcut = x
        x = self._conv_layer(x, name=stage + 'conv1', strides=strides)

        x = self._get_batchnorm_layer(x, name=stage + 'bn2')
        x = self._get_relu_activation(x, name=stage + 'relu2')
        x = self._conv_layer(x, name=stage + 'conv2')

        shortcut = self._conv_layer(
            shortcut, name=stage + 'sc', strides=strides)
        x = tf.add(x, shortcut, name=stage + 'add')

        return x

    #-----------------------#
    # identity block        #
    #-----------------------#
    def _res_identity_block(self, input_layer, stage):
        x = self._get_batchnorm_layer(input_layer, name=stage + 'bn1')
        x = self._get_relu_activation(x, name=stage + 'relu1')
        x = self._conv_layer(x, name=stage + 'conv1')

        x = self._get_batchnorm_layer(x, name=stage + 'bn2')
        x = self._get_relu_activation(x, name=stage + 'relu2')
        x = self._conv_layer(x, name=stage + 'conv2')

        x = tf.add(x, input_layer, name=stage + 'add')

        return x
