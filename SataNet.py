from keras.layers import Conv2D, SeparableConv2D
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import Add, Concatenate
from keras.layers import Dropout
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Softmax
from keras import backend as K


class SataNet():
    def __init__(self, height=32, width=32, depth=3, classes=10):
        self.use_bias = False
        self.padding = "same"
        self.scale = False
        self.input_shape = (height, width, depth)
        self.classes = classes

        self.activation = "prelu"
        self.weight_reg_factor = 0.
        self.use_dropout = True
        self.pool_size = 3
        self.pool_stride = 2

        self.filters_0 = 24
        self.kernel_size_0 = 5
        self.downsample_0 = False
        self.dropout_rate_0 = 1 / 1024

        self.filters_1 = 24
        self.kernel_size_0 = 5
        self.downsample_0 = False
        self.dropout_rate_0 = 1 / 1024

    def _activation(self, x, name):
        activation_name = name + self.activation
        if (self.activation == "leaky"):
            x = LeakyReLU(name=activation_name)(x)
        elif (self.activation == "prelu"):
            x = PReLU(shared_axes=[1, 2], name=activation_name)(x)
        else:
            x = Activation(self.activation, name=activation_name)(x)
        return x

    def _binary_reg(self, weight_matrix):
        return self.weight_reg_factor * K.sum(
            K.abs(K.abs(weight_matrix) - 1.0))

    def _batchNormAndActivation(self, x_input, name):
        x = BatchNormalization(scale=self.scale, name=name + "bn")(x_input)
        return self._activation(x, name=name)

    def _first_conv_block(self, x, filters, kernel_size, downsample,
                          dropout_rate, name):
        layer_name = name + "_" + str(kernel_size) + "x" + str(kernel_size)

        x = Conv2D(
            filters,
            kernel_size=kernel_size,
            kernel_regularizer=self._binary_reg,
            use_bias=self.use_bias,
            padding=self.padding,
            input_shape=self.input_shape,
            name=layer_name + "conv")(x)
        x = self._batchNormAndActivation(x, name=name)
        if downsample:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding,
                name=name + "maxpool")(x)
        if self.use_dropout:
            x = Dropout(rate=dropout_rate, name=name + "dropout")(x)
        return x

    def _conv_block(self,
                    x_input,
                    filters,
                    kernel_size,
                    name,
                    dilation_rate=1,
                    batch_norm=False,
                    activation=False):
        layer_name = name + "_" + str(kernel_size) + "x" + str(kernel_size)
        if kernel_size == 1:
            x = Conv2D(
                filters,
                kernel_size=kernel_size,
                kernel_regularizer=self._binary_reg,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name + "conv")(x_input)
        else:
            x = SeparableConv2D(
                filters,
                kernel_size=kernel_size,
                depthwise_regularizer=self._binary_reg,
                pointwise_regularizer=self._binary_reg,
                dilation_rate=dilation_rate,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name + "conv")(x_input)
        if batch_norm:
            x = BatchNormalization(scale=self.scale, name=layer_name + "bn")(x)
        if activation:
            x = self._activation(x, name=layer_name)
        return x

    def _module(self, x_input, filters, kernel_size, downsample, dropout_rate,
                name):
        num_branches = (kernel_size + 1) // 2
        filters_per_branch = filters // num_branches

        if num_branches == 1:
            x = self._conv_block(
                x_input,
                filters=filters_per_branch,
                kernel_size=kernel_size,
                name=name + "d0")
        else:
            tensor_list = []
            for n in range(num_branches):
                x = self._conv_block(
                    x_input,
                    filters=filters_per_branch,
                    kernel_size=kernel_size - 2 * n,
                    name=name + "d" + str(n))
                tensor_list.append(x)
            x = Concatenate(name=name + "concat")(tensor_list)
        x = self._batchNormAndActivation(x, name=name)

        if downsample:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding,
                name=name + "maxpool")(x)
        if self.use_dropout:
            x = Dropout(rate=dropout_rate, name=name + "dropout")(x)
        return x

    def build(self, x):
        x = self._first_conv_block(
            x,
            self.filters_0,
            self.kernel_size_0,
            self.downsample_0,
            self.dropout_rate_0,
            name="first")
        for n in range(self.num_blocks):
            module_name = "m" + str(n) + "_"
            x = self._module(
                x,
                self.filters[n],
                self.kernel_size[n],
                self.downsample[n],
                self.dropout_rate[n],
                name=module_name)
        x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = Softmax(name="softmax")(x)
        return x
