# -*- coding: utf-8 -*-
"""Utils for binary layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Dropout
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils


class _Quantization():
    def __init__(self, num_bits=8, minval=-1, maxval=1):
        self.num_bits = num_bits
        self.minval = minval
        self.maxval = maxval
        self.scale = (2**num_bits - 1) / (self.maxval - self.minval)
        self.inv_scale = 1 / self.scale

    def __call__(self, x):
        x_clip = K.clip(x, self.minval, self.maxval)
        x_scale = (x_clip - self.minval) * self.scale
        forward_behavior = self.inv_scale * K.round(x_scale) + self.minval
        backward_behavior = x_clip
        # Trick by Sergey Ioffe apparently.
        return backward_behavior + K.stop_gradient(forward_behavior -
                                                   backward_behavior)


class Quantization(Layer):
    def __init__(self, num_bits=8, **kwargs):
        super(Quantization, self).__init__(**kwargs)
        self.supports_masking = True
        self.num_bits = num_bits
        self.activation = _Quantization(num_bits=self.num_bits)

    def call(self, inputs):
        return self.activation(inputs)

    def get_config(self):
        config = {}
        base_config = super(Quantization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class QuantizationRegularizer(regularizers.Regularizer):
    def __init__(self, scale=0.):
        self.scale = K.cast_to_floatx(scale)

    def __call__(self, x):
        if self.scale:
            regularization = K.sum(
                self.scale * K.square(K.relu(K.abs(x) - 1.)))
        else:
            regularization = 0.
        return regularization

    def get_config(self):
        return {"scale": float(self.scale)}


class QuantizedConv2D(Layer):
    def __init__(self,
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
                 num_bits=8,
                 **kwargs):
        super(QuantizedConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank,
                                                  'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.num_bits = num_bits
        self.quantization = _Quantization(num_bits=self.num_bits)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError("The channel dimension of the inputs "
                             "should be defined. Found `None`.")
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
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

    def call(self, inputs):
        # Quantize the kernel.
        quantized_kernel = self.quantization(self.kernel)

        outputs = K.conv2d(
            inputs,
            quantized_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
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
        if self.data_format == "channels_first":
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
            'num_bits':
            self.num_bits
        }
        base_config = super(QuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantizedDense(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 num_bits=8,
                 **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"), )
        super(QuantizedDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.num_bits = num_bits
        self.quantization = _Quantization(num_bits=self.num_bits)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units, ),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        # Quantize the kernel.
        quantized_kernel = self.quantization(self.kernel)
        output = K.dot(inputs, quantized_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            "units":
            self.units,
            "activation":
            activations.serialize(self.activation),
            "kernel_initializer":
            initializers.serialize(self.kernel_initializer),
            "kernel_regularizer":
            regularizers.serialize(self.kernel_regularizer),
            "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
            "kernel_constraint":
            constraints.serialize(self.kernel_constraint),
            'num_bits':
            num_bits
        }
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
