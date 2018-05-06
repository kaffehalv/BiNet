# -*- coding: utf-8 -*-
"""Convolutional layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces

EPSILON = 1e-3
STDDEV = 0.0


def binarization(x,
                 epsilon=EPSILON,
                 stddev=STDDEV,
                 test_hard=True,
                 training=None):
    def soft_binary():
        # Add noise.
        if stddev > 0.0:
            x_noise = x + K.random_normal(
                shape=K.shape(x), mean=0., stddev=stddev)
        else:
            x_noise = x
        # The factor (1 + eps) is to ensure that abs(x)=1 returns x.
        return x_noise * (1.0 + epsilon) / (K.abs(x_noise) + epsilon)

    def hard_binary():
        return K.sign(x)

    if test_hard:
        return K.in_train_phase(soft_binary, hard_binary, training=training)
    else:
        return soft_binary()


class _Conv(Layer):
    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_epsilon=EPSILON,
                 kernel_noise_stddev=STDDEV,
                 test_hard=True,
                 **kwargs):
        super(_Conv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.kernel_epsilon = kernel_epsilon
        self.kernel_noise_stddev = kernel_noise_stddev
        self.test_hard = test_hard

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters, ),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        #self.kernel = K.clip(self.kernel, -1.0, 1.0)
        binary_kernel = binarization(
            self.kernel,
            epsilon=self.kernel_epsilon,
            stddev=self.kernel_noise_stddev,
            test_hard=self.test_hard,
            training=training)
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                binary_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            raise ValueError("No fucking bias!")

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], ) + tuple(new_space) + (self.filters, )
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank':
            self.rank,
            'filters':
            self.filters,
            'kernel_size':
            self.kernel_size,
            'strides':
            self.strides,
            'padding':
            self.padding,
            'data_format':
            self.data_format,
            'dilation_rate':
            self.dilation_rate,
            'activation':
            activations.serialize(self.activation),
            'use_bias':
            self.use_bias,
            'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
            'bias_initializer':
            initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
            'bias_constraint':
            constraints.serialize(self.bias_constraint),
            'kernel_epsilon':
            self.kernel_epsilon
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinaryConv2D(_Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(BinaryConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        return config


class Binarization(Layer):
    def __init__(self,
                 epsilon=EPSILON,
                 stddev=STDDEV,
                 test_hard=True,
                 activity_regularizer=None,
                 **kwargs):
        super(Binarization, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = binarization
        self.epsilon = epsilon
        self.stddev = stddev
        self.test_hard = test_hard
        self.activity_regularizer = activity_regularizer

    def call(self, inputs, training=None):
        return self.activation(
            inputs,
            epsilon=self.epsilon,
            stddev=self.stddev,
            test_hard=self.test_hard,
            training=training)

    def get_config(self):
        config = {
            'epsilon':
            self.epsilon,
            'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        }
        base_config = super(Binarization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
