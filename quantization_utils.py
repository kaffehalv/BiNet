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
        return backward_behavior + K.stop_gradient(
            forward_behavior - backward_behavior)


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


class QuantizedActivityRegularization(Layer):
    def __init__(self, scale=0., **kwargs):
        super(QuantizedActivityRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.scale = scale
        self.activity_regularizer = QuantizationRegularizer(scale=self.scale)

    def get_config(self):
        config = {"scale": self.scale}
        base_config = super(QuantizedActivityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class QuantizedConv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding="valid",
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 scale_kernel=False,
                 num_bits=8,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(QuantizedConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank,
                                                      "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, self.rank,
                                                  "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, self.rank, "dilation_rate")
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.scale_kernel = scale_kernel
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

        # Set input spec.
        self.input_spec = InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Quantize the kernel.
        quantized_kernel = self.quantization(self.kernel)

        if self.scale_kernel:
            quantized_kernel = quantized_kernel * K.mean(
                K.abs(self.kernel), axis=[0, 1, 2], keepdims=True)

        outputs = K.conv2d(
            inputs,
            quantized_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

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
            "filters":
            self.filters,
            "kernel_size":
            self.kernel_size,
            "strides":
            self.strides,
            "padding":
            self.padding,
            "data_format":
            self.data_format,
            "dilation_rate":
            self.dilation_rate,
            "activation":
            activations.serialize(self.activation),
            "kernel_initializer":
            initializers.serialize(self.kernel_initializer),
            "kernel_regularizer":
            regularizers.serialize(self.kernel_regularizer),
            "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
            "kernel_constraint":
            constraints.serialize(self.kernel_constraint)
        }
        base_config = super(QuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantizedDepthwiseConv2D(Layer):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 depth_multiplier=1,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 scale_kernel=False,
                 num_bits=8,
                 depthwise_initializer="glorot_uniform",
                 depthwise_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 **kwargs):
        super(QuantizedDepthwiseConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank,
                                                      "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, self.rank,
                                                  "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, self.rank, "dilation_rate")
        self.activation = activations.get(activation)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.scale_kernel = scale_kernel
        self.num_bits = num_bits
        self.quantization = _Quantization(num_bits=self.num_bits)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError(
                "Inputs to `QuantizedDepthwiseConv2D` should have rank 4. "
                "Received input shape:", str(input_shape))
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError("The channel dimension of the inputs to "
                             "`QuantizedDepthwiseConv2D` "
                             "should be defined. Found `None`.")
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                                  input_dim, self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name="depthwise_kernel",
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Quantize the kernel.
        quantized_kernel = self.quantization(self.depthwise_kernel)

        if self.scale_kernel:
            quantized_kernel = quantized_kernel * K.mean(
                K.abs(self.depthwise_kernel), axis=[0, 1, 2], keepdims=True)

        outputs = K.depthwise_conv2d(
            inputs,
            quantized_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == "channels_last":
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding, self.strides[1])
        if self.data_format == "channels_first":
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == "channels_last":
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = {
            "filters":
            self.filters,
            "kernel_size":
            self.kernel_size,
            "strides":
            self.strides,
            "padding":
            self.padding,
            "data_format":
            self.data_format,
            "dilation_rate":
            self.dilation_rate,
            "activation":
            activations.serialize(self.activation),
            "depthwise_initializer":
            initializers.serialize(self.depthwise_initializer),
            "depthwise_regularizer":
            regularizers.serialize(self.depthwise_regularizer),
            "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
            "depthwise_constraint":
            constraints.serialize(self.depthwise_constraint)
        }
        base_config = super(QuantizedDepthwiseConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantizedDense(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 scale_kernel=False,
                 num_bits=8,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"), )
        super(QuantizedDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.scale_kernel = scale_kernel
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

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        # Quantize the kernel.
        quantized_kernel = self.quantization(self.kernel)

        if self.scale_kernel:
            quantized_kernel = quantized_kernel * K.mean(K.abs(self.kernel))
        output = K.dot(inputs, quantized_kernel)

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
        }
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))