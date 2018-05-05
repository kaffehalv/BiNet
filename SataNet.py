from keras.layers import Conv2D, BatchNormalization, Activation, PReLU
from keras.layers import Dropout, GaussianDropout, Add, SeparableConv2D
from keras.layers import AveragePooling2D, Concatenate
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Softmax
from keras import backend as K


class SataNet():
    def __init__(self, height=32, width=32, depth=3, classes=10):
        self.use_bias = False
        self.scale = True
        self.input_shape = (height, width, depth)
        self.classes = classes

        self.activation = "prelu"
        self.pool_size = 3
        self.pool_stride = 2
        self.padding = "same"
        self.conv_type = "full"

        self.dropout_rate = 0.1
        self.gaussian_dropout_rate = 0.1
        self.weight_reg_factor = 1e-5

        self.filters_0 = 16
        self.kernel_size_0 = 3

        self.filters_1 = 32
        self.kernel_size_1 = 3
        self.repeats_1 = 2

        self.filters_2 = 64
        self.kernel_size_2 = 3
        self.repeats_2 = 2

        self.filters_3 = 128
        self.kernel_size_3 = 3
        self.repeats_3 = 1

        self.block_id = 0

    def _binary_reg(self, weight_matrix):
        return self.weight_reg_factor * K.sum(
            K.square(K.abs(weight_matrix) - 1.0))

    def _activation(self, x, name, is_dense=False):
        activation_name = name + "_" + self.activation
        if (self.activation == "prelu"):
            if is_dense:
                x = PReLU(name=activation_name)(x)
            else:
                x = PReLU(shared_axes=[1, 2], name=activation_name)(x)
        else:
            x = Activation(self.activation, name=activation_name)(x)
        return x

    def _batch_norm(self, x_input, name):
        return BatchNormalization(scale=self.scale, name=name + "_bn")(x_input)

    def _batchNormAndActivation(self, x_input, name, is_dense=False):
        x = self._batch_norm(x_input, name=name)
        return self._activation(x, name=name, is_dense=is_dense)

    def _conv_block(self,
                    x,
                    filters,
                    kernel_size,
                    name,
                    do_act=True,
                    strides=1):
        if type(kernel_size) is tuple:
            kh = kernel_size[0]
            kw = kernel_size[1]
        else:
            kh = kernel_size
            kw = kernel_size
        layer_name = name + "_conv" + str(kh) + "x" + str(kw)
        x = Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_regularizer=self._binary_reg,
            use_bias=self.use_bias,
            padding=self.padding,
            name=layer_name)(x)
        if self.gaussian_dropout_rate > 0.0:
            x = GaussianDropout(
                rate=self.gaussian_dropout_rate, name=layer_name + "_gauss")(x)
        x = self._batch_norm(x, name=layer_name)
        if do_act:
            x = self._activation(x, name=layer_name)
        return x

    def _dw_conv_block(self,
                       x,
                       filters,
                       kernel_size,
                       name,
                       do_act=True,
                       strides=1):
        if type(kernel_size) is tuple:
            kh = kernel_size[0]
            kw = kernel_size[1]
        else:
            kh = kernel_size
            kw = kernel_size
        layer_name = name + "_conv" + str(kh) + "x" + str(kw)
        x = SeparableConv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            depthwise_regularizer=self._binary_reg,
            pointwise_regularizer=self._binary_reg,
            use_bias=self.use_bias,
            padding=self.padding,
            name=layer_name)(x)
        if self.gaussian_dropout_rate > 0.0:
            x = GaussianDropout(
                rate=self.gaussian_dropout_rate, name=layer_name + "_gauss")(x)
        x = self._batch_norm(x, name=layer_name)
        if do_act:
            x = self._activation(x, name=layer_name)
        return x

    def _unit(self, x_in, filters, kernel_size, stride, name):
        if self.conv_type == "full":
            x = self._conv_block(
                x_in,
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                name=name,
                do_act=False)
        elif self.conv_type == "dw_sep":
            x = self._dw_conv_block(
                x_in,
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                name=name,
                do_act=False)

        if stride == 1:
            merge_name = name + "_add"
            x = Add(name=merge_name)([x_in, x])
        else:
            merge_name = name + "_concat"
            x_pool = AveragePooling2D(name=name + "_pool")(x_in)
            x = Concatenate(name=merge_name)([x_pool, x])
        x = self._activation(x, name=merge_name)
        return x

    def _module(self, x, filters, kernel_size, repeats):
        self.block_id += 1
        name = "m" + str(self.block_id)
        for n in range(repeats):
            unit_name = name + "_b" + str(n)
            if n == 0:
                x = self._unit(
                    x,
                    filters=filters // 2,
                    kernel_size=kernel_size,
                    stride=2,
                    name=unit_name)
            else:
                x = self._unit(
                    x,
                    filters=filters,
                    kernel_size=kernel_size,
                    stride=1,
                    name=unit_name)
        return x

    def build(self, x):
        x = self._conv_block(
            x,
            filters=self.filters_0,
            kernel_size=self.kernel_size_0,
            name="first")

        x = self._module(
            x,
            filters=self.filters_1,
            kernel_size=self.kernel_size_1,
            repeats=self.repeats_1)

        x = self._module(
            x,
            filters=self.filters_2,
            kernel_size=self.kernel_size_2,
            repeats=self.repeats_2)

        x = self._module(
            x,
            filters=self.filters_3,
            kernel_size=self.kernel_size_3,
            repeats=self.repeats_3)

        x = GlobalAveragePooling2D(name="global_avg_pool")(x)

        if self.dropout_rate > 0.0:
            x = Dropout(rate=self.dropout_rate, name="dropout")(x)

        x = Dense(self.classes, name="dense")(x)
        x = Softmax(name="softmax")(x)
        return x
