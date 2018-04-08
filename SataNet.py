from keras.layers import Conv2D
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import Add, Concatenate
from keras.layers import Dropout
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Softmax


class SataNet():
    def __init__(self, height=32, width=32, depth=3, classes=10):
        self.use_bias = False
        self.padding = "same"
        self.scale = False
        self.input_shape = (height, width, depth)
        self.classes = classes

        self.activation = "leaky"
        self.dropout_rate = 0.5
        self.dw_conv_size = 3
        self.pool_size = 3
        self.pool_stride = 2

        self.first_filters = 32
        self.first_kernel_size = 3
        self.first_downsample = False

        self.squeeze_1 = 16
        self.expand_1 = 64
        self.repeat_1 = 2
        self.downsample_1 = True

        self.squeeze_2 = 32
        self.expand_2 = 128
        self.repeat_2 = 2
        self.downsample_2 = True

        self.squeeze_3 = 64
        self.expand_3 = 256
        self.repeat_3 = 2
        self.downsample_3 = True

        self.module_number = 0

    def _activation(self, x, name):
        activation_name = name + "_" + self.activation
        if (self.activation == "leaky"):
            x = LeakyReLU(name=activation_name)(x)
        elif (self.activation == "prelu"):
            x = PReLU(v)(x)
        else:
            x = Activation(self.activation, name=activation_name)(x)
        return x

    def _batchNormAndActivation(self, x_input, name):
        x = BatchNormalization(scale=self.scale, name=name + "_bn")(x_input)
        return self._activation(x, name=name)

    def _pointwiseConv2D(self,
                         x_input,
                         filters,
                         name,
                         batch_norm=True,
                         activation=True):
        x = Conv2D(
            filters, (1, 1),
            use_bias=self.use_bias,
            padding=self.padding,
            name=name + "_1x1")(x_input)
        if batch_norm:
            x = BatchNormalization(scale=self.scale, name=name + "_bn")(x)
        if activation:
            x = self._activation(x, name=name)
        return x

    def _depthwiseConv2D(self,
                         x_input,
                         kernel_size,
                         name,
                         batch_norm=True,
                         activation=True):
        x = DepthwiseConv2D(
            (kernel_size, 1),
            use_bias=self.use_bias,
            padding=self.padding,
            name=name + "_" + str(kernel_size) + "x1")(x_input)
        x = DepthwiseConv2D(
            (1, kernel_size),
            use_bias=self.use_bias,
            padding=self.padding,
            name=name + "_1x" + str(kernel_size))(x)
        if batch_norm:
            x = BatchNormalization(scale=self.scale, name=name + "_bn")(x)
        if activation:
            x = self._activation(x, name=name)
        return x

    def _module(self, x_input, kernel_size, squeeze, expand, name):
        x_left = self._pointwiseConv2D(x_input, expand, name=name + "_left_pw")

        x_right = self._pointwiseConv2D(
            x_input, squeeze, name=name + "_right_squeeze_pw")
        x_right = self._depthwiseConv2D(
            x_right, kernel_size, name=name + "_right_dw")
        x_right = self._pointwiseConv2D(
            x_right, expand, name=name + "_right_expand_pw")

        x = Add(name=name + "_add")([x_left, x_right])
        return self._batchNormAndActivation(x, name=name)

    def _module_repeat(self, x_input, kernel_size, squeeze, expand, repeat):
        for _ in range(repeat):
            self.module_number += 1
            name = "m" + str(self.module_number)
            x = self._module(x_input, kernel_size, squeeze, expand, name)
        return x

    def build(self, x_input):
        x = Conv2D(
            self.first_filters,
            self.first_kernel_size,
            input_shape=self.input_shape,
            use_bias=self.use_bias,
            padding=self.padding,
            name="first_" + str(self.first_kernel_size) + "x" +
            str(self.first_kernel_size))(x_input)
        x = self._batchNormAndActivation(x, name="first")
        if self.first_downsample:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding)(x)

        x = self._module_repeat(x, self.dw_conv_size, self.squeeze_1,
                                self.expand_1, self.repeat_1)
        if self.downsample_1:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding)(x)

        x = self._module_repeat(x, self.dw_conv_size, self.squeeze_2,
                                self.expand_2, self.repeat_2)
        if self.downsample_2:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding)(x)

        x = self._module_repeat(x, self.dw_conv_size, self.squeeze_3,
                                self.expand_3, self.repeat_3)
        if self.downsample_3:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding)(x)

        x = Dropout(rate=self.dropout_rate)(x)
        x = self._pointwiseConv2D(x, self.classes, name="last")
        x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = Softmax(name="softmax")(x)
        return x
