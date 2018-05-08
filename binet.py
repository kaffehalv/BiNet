from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Dropout
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Softmax
from keras import backend as K
from binet_utils import BinaryConv2D, BinaryDepthwiseConv2D, Binarization, BinaryRegularizer


class BiNet():
    def __init__(self,
                 conv_type="binary",
                 activation="binary",
                 weight_reg_strength=1e-6,
                 activity_reg_strength=1e-6,
                 dropout_rate=0.0,
                 input_shape=(32, 32, 3),
                 classes=10):
        self.conv_type = conv_type
        self.activation = activation
        self.weight_reg_strength = weight_reg_strength
        self.activity_reg_strength = activity_reg_strength
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.classes = classes

        # Conv bias unnecessary.
        self.use_bias = False

        # Batch norm scale unnecessary with binarization.
        self.scale = False

        ####### Network Architecture #######
        self.filters_1 = 32
        self.repeats_1 = 2

        self.filters_2 = 64
        self.repeats_2 = 2

        self.filters_3 = 128
        self.repeats_3 = 2

        self.pool_size = 2
        self.pool_stride = 2
        self.padding = "same"
        ####################################

        # Just for naming layers.
        self.module_id = 0

    def _activation(self, x, name, is_dense=False):
        activation_name = name + "_" + self.activation
        if (self.activation == "binary"):
            x = Binarization(
                activity_regularizer=BinaryRegularizer(
                    self.activity_reg_strength),
                name=activation_name)(x)
        elif (self.activation == "prelu"):
            if is_dense:
                x = PReLU(name=activation_name)(x)
            else:
                x = PReLU(shared_axes=[1, 2], name=activation_name)(x)
        else:
            x = Activation(self.activation, name=activation_name)(x)
        return x

    def _conv_block(self, x, filters, pool, name):
        if self.conv_type == "float":
            layer_name = name
            x = Conv2D(
                filters,
                kernel_size=3,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name + "_conv")(x)
        elif self.conv_type == "float_dw":
            layer_name = name + "_dw"
            x = DepthwiseConv2D(
                kernel_size=3,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name+ "_conv")(x)
            x = self._batch_norm(x, name=layer_name)
            x = self._activation(x, name=layer_name)
            layer_name = name + "_pw"
            x = Conv2D(
                filters,
                kernel_size=1,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name + "_conv")(x)
        elif self.conv_type == "binary":
            layer_name = name
            x = BinaryConv2D(
                filters,
                kernel_size=3,
                kernel_regularizer=BinaryRegularizer(self.weight_reg_strength),
                padding=self.padding,
                name=layer_name+ "_conv")(x)
        elif self.conv_type == "binary_dw":
            layer_name = name + "_dw"
            x = BinaryDepthwiseConv2D(
                kernel_size=3,
                depthwise_regularizer=BinaryRegularizer(self.weight_reg_strength),
                padding=self.padding,
                name=layer_name+ "_conv")(x)
            x = self._batch_norm(x, name=layer_name)
            x = self._activation(x, name=layer_name)
            layer_name = name + "_pw"
            x = BinaryConv2D(
                filters,
                kernel_size=1,
                kernel_regularizer=BinaryRegularizer(self.weight_reg_strength),
                padding=self.padding,
                name=layer_name + "_conv")(x)
        if pool:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                name=name + "_maxpool")(x)
        x = self._batch_norm(x, name=layer_name)
        x = self._activation(x, name=layer_name)
        return x

    def _batch_norm(self, x, name):
        return BatchNormalization(scale=self.scale, name=name + "_bn")(x)

    def _module(self, x, filters, repeats):
        self.module_id += 1
        name = "m" + str(self.module_id)
        for n in range(repeats):
            block_name = name + "_b" + str(n)
            if n == (repeats - 1):
                x = self._conv_block(
                    x, filters=filters, pool=True, name=block_name)
            else:
                x = self._conv_block(
                    x, filters=filters, pool=False, name=block_name)
        return x

    def build(self, x):
        x = self._module(x, filters=self.filters_1, repeats=self.repeats_1)

        x = self._module(x, filters=self.filters_2, repeats=self.repeats_2)

        x = self._module(x, filters=self.filters_3, repeats=self.repeats_3)

        x = GlobalAveragePooling2D(name="global_avg_pool")(x)

        if self.dropout_rate > 0.0:
            x = Dropout(rate=self.dropout_rate, name="dropout")(x)

        x = Dense(self.classes, name="dense")(x)
        x = Softmax(name="softmax")(x)
        return x
