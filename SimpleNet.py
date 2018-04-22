from keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import Activation, PReLU
from keras.layers import Dropout
from keras.layers import MaxPooling2D
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
        self.xy_separation_limit = 7

        self.activation = "prelu"
        self.factorize_3x3 = False
        self.use_dropout = True
        self.pool_size = 3
        self.pool_stride = 2
        self.padding = "same"

        self.filters = [24, 48, 48, 48, 96, 96, 96,192, 192, self.classes]
        self.kernel_size = [3, 7, 7,7, 3, 3,3, 3, 3, 1]
        self.conv_type = [
            "full", "sep", "sep", "sep", "sep", "sep", "sep","sep", "sep", "full"
        ]
        self.downsample = [False, False, False, True, False, False, True, False, True, False]
        #self.dropout_rate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
        self.dropout_rate = [1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 0.0]
        self.num_blocks = len(self.filters)

        self.dense_neurons = 128

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

    def _full_conv_block(self, x, filters, kernel_size, name, dilation_rate=1):
        if kernel_size < self.xy_separation_limit:
            layer_name = name + "_conv" + str(kernel_size) + "x" + str(
                kernel_size)
            x = Conv2D(
                filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name)(x)
            x = self._batchNormAndActivation(x, name=layer_name)
        else:
            layer_name = name + "_conv1x" + str(kernel_size)
            x = Conv2D(
                x.shape[-1],
                kernel_size=(1, kernel_size),
                dilation_rate=dilation_rate,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name)(x)
            x = self._batchNormAndActivation(x, name=layer_name)

            layer_name = name + "_conv" + str(kernel_size) + "x1"
            x = Conv2D(
                filters,
                kernel_size=(kernel_size, 1),
                dilation_rate=dilation_rate,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name)(x)
            x = self._batchNormAndActivation(x, name=layer_name)
        return x

    def _sep_conv_block(self, x, filters, kernel_size, name, dilation_rate=1):
        if kernel_size < self.xy_separation_limit:
            if self.factorize_3x3 and kernel_size == 3:
                layer_name = name + "_dw2x2_A"
                x = DepthwiseConv2D(
                    kernel_size=2,
                    dilation_rate=dilation_rate,
                    use_bias=self.use_bias,
                    padding=self.padding,
                    name=layer_name)(x)
                x = self._batchNormAndActivation(x, name=layer_name)

                layer_name = name + "_dw2x2_B"
                x = DepthwiseConv2D(
                    kernel_size=2,
                    dilation_rate=dilation_rate,
                    use_bias=self.use_bias,
                    padding=self.padding,
                    name=layer_name)(x)
                x = self._batchNormAndActivation(x, name=layer_name)
            else:
                layer_name = name + "_dw" + str(kernel_size) + "x" + str(
                    kernel_size)
                x = DepthwiseConv2D(
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    use_bias=self.use_bias,
                    padding=self.padding,
                    name=layer_name)(x)
                x = self._batchNormAndActivation(x, name=layer_name)
        else:
            layer_name = name + "_dw1x" + str(kernel_size)
            x = DepthwiseConv2D(
                kernel_size=(1, kernel_size),
                dilation_rate=dilation_rate,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name)(x)
            x = self._batchNormAndActivation(x, name=layer_name)

            layer_name = name + "_dw" + str(kernel_size) + "x1"
            x = DepthwiseConv2D(
                kernel_size=(kernel_size, 1),
                dilation_rate=dilation_rate,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name)(x)
            x = self._batchNormAndActivation(x, name=layer_name)

        layer_name = name + "_pw1x1"
        x = Conv2D(
            filters,
            kernel_size=1,
            dilation_rate=dilation_rate,
            use_bias=self.use_bias,
            padding=self.padding,
            input_shape=self.input_shape,
            name=layer_name)(x)
        x = self._batchNormAndActivation(x, name=layer_name)
        return x

    def _conv_block(self,
                    x,
                    filters,
                    kernel_size,
                    name,
                    conv_type,
                    downsample=False,
                    dropout_rate=1e-3,
                    dilation_rate=1,
                    is_first_layer=False):

        if conv_type == "full":
            x = self._full_conv_block(
                x,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                name=name)
        else:
            x = self._sep_conv_block(
                x,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                name=name)

        if downsample:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding,
                name=name + "_maxpool")(x)

        if self.use_dropout and (dropout_rate > 0.0):
            x = Dropout(rate=dropout_rate, name=name + "_dropout")(x)

        return x

    def build(self, x):
        for n in range(self.num_blocks):
            name = "b" + str(n)
            is_first_layer = n == 0
            x = self._conv_block(
                x,
                filters=self.filters[n],
                kernel_size=self.kernel_size[n],
                conv_type=self.conv_type[n],
                downsample=self.downsample[n],
                dropout_rate=self.dropout_rate[n],
                name=name,
                is_first_layer=is_first_layer)

        layer_name = "global_avg_pool"
        x = GlobalAveragePooling2D(name=layer_name)(x)
        x = Softmax(name="softmax")(x)
        return x
