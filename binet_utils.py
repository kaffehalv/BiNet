# -*- coding: utf-8 -*-
"""Utils for binary layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces


def _binarization(x, training=None):
    def binary():
        # This method is only here to be sure that the binarization
        # behaves like expected. Technically, it is completely unnecessary.
        return K.sign(x)

    def sneaky_fucker_binary():
        # Trick by Sergey Ioffe apparently.
        forward_behavior = K.sign(x)
        backward_behavior = K.clip(x, -1., 1.)
        return backward_behavior + K.stop_gradient(
            forward_behavior - backward_behavior)

    return K.in_train_phase(sneaky_fucker_binary, binary, training=training)


class Binarization(Layer):
    def __init__(self, activity_regularizer=None, **kwargs):
        super(Binarization, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = _binarization
        self.activity_regularizer = activity_regularizer

    def call(self, inputs, training=None):
        return self.activation(inputs, training=training)

    def get_config(self):
        config = {
            'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        }
        base_config = super(Binarization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class BinaryRegularizer(regularizers.Regularizer):
    def __init__(self, scale=0.):
        self.scale = K.cast_to_floatx(scale)

    def __call__(self, x):
        if self.scale:
            x2 = K.square(x)
            regularization = K.sum(self.scale * (K.square(x2) - 2. * x2 + 1.))
        else:
            regularization = 0.
        return regularization

    def get_config(self):
        return {'scale': float(self.scale)}


class BinaryConv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(BinaryConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank,
                                                  'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

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

        # Set input spec.
        self.input_spec = InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        # Binarize the kernel.
        binary_kernel = _binarization(self.kernel, training=training)

        outputs = K.conv2d(
            inputs,
            binary_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

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
            'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
            'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
            constraints.serialize(self.kernel_constraint)
        }
        base_config = super(BinaryConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
