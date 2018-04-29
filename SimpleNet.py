from keras.layers import Conv2D, BatchNormalization, Activation, PReLU
from keras.layers import Dropout, Add, DepthwiseConv2D
from keras.layers import MaxPooling2D, Concatenate
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Softmax
from keras import backend as K


class SimpleNet():
    def __init__(self, height=32, width=32, depth=3, classes=10):
        self.use_bias = False
        self.scale = False
        self.input_shape = (height, width, depth)
        self.classes = classes

        self.activation = "prelu"
        self.use_dropout = True
        self.pool_size = 3
        self.pool_stride = 2
        self.padding = "same"

        self.filters_0 = 32
        self.kernel_size_0 = 3
        self.downsample_0 = True

        self.filters_1 = 64
        self.repeats_1 = 2
        self.downsample_1 = True

        self.filters_2 = 128
        self.repeats_2 = 2
        self.downsample_2 = False

        self.dense_neurons = 64

        self.block_id = 0

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

    def _batchNormAndActivation(self, x_input, name, is_dense=False):
        x = BatchNormalization(scale=self.scale, name=name + "_bn")(x_input)
        return self._activation(x, name=name, is_dense=is_dense)

    def _conv_block(self, x, filters, kernel_size, name, strides=1):
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
            use_bias=self.use_bias,
            padding=self.padding,
            name=layer_name)(x)
        x = self._batchNormAndActivation(x, name=layer_name)
        return x

    def _squeezenext_unit(self, x_in, filters, stride, name):
        x = self._conv_block(
            x_in, filters=filters // 2, kernel_size=1, name=name + "_squeeze1")
        x = self._conv_block(
            x, filters=filters // 4, kernel_size=1, name=name + "_squeeze2")
        x = self._conv_block(
            x, filters=filters // 2, kernel_size=(1, 3), strides=(1, stride), name=name)
        x = self._conv_block(
            x, filters=filters // 2, kernel_size=(3, 1), strides=(stride, 1), name=name)
        x = self._conv_block(
            x, filters=filters, kernel_size=1, name=name + "_expand")

        if stride == 1:
            merge_name = name + "_add"
            x = Add(name=merge_name)([x_in, x])
        else:
            merge_name = name + "_concat"
            x_pool = MaxPooling2D(name=name+"_pool")(x_in)
            x = Concatenate(name=merge_name)([x_pool, x])
        x = self._batchNormAndActivation(x, name=merge_name)
        return x

    def _squeezenext_block(self, x, filters, repeats):
        self.block_id += 1
        name = "b" + str(self.block_id)
        for n in range(repeats):
            unit_name = name + "_u" + str(n)
            if n == 0:
                stride = 2
                f = filters // 2
            else:
                stride = 1
                f = filters
            x = self._squeezenext_unit(x, filters=f, stride=stride, name=unit_name)
        return x

    def build(self, x):
        x = self._conv_block(
            x,
            filters=self.filters_0,
            kernel_size=self.kernel_size_0,
            name="first")

        x = self._squeezenext_block(
            x, filters=self.filters_1, repeats=self.repeats_1)

        x = self._squeezenext_block(
            x, filters=self.filters_2, repeats=self.repeats_2)

        x = self._conv_block(
            x, filters=self.dense_neurons, kernel_size=1, name="last")

        layer_name = "global_avg_pool"
        x = GlobalAveragePooling2D(name=layer_name)(x)
        x = Dense(self.classes)(x)
        x = Softmax(name="softmax")(x)
        return x
