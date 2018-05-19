from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, SpatialDropout2D
from keras.layers import MaxPooling2D, Flatten, Dropout
from keras.layers import Dense, Softmax
from keras import backend as K
from binet_utils import BinaryConv2D, BinaryDense, BinaryDepthwiseConv2D
from binet_utils import Binarization, BinaryActivityRegularization, BinaryRegularizer


class BiNet():
    def __init__(self,
                 weight_type="binary",
                 activation="binary",
                 shrink=1,
                 weight_reg_strength=0.0,
                 activity_reg_strength=0.0,
                 dropout_rate=0.0,
                 input_shape=(32, 32, 3),
                 classes=10):
        self.weight_type = weight_type
        self.activation = activation
        self.weight_reg_strength = weight_reg_strength
        self.activity_reg_strength = activity_reg_strength
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.classes = classes

        # Set up the regularizers.
        if self.weight_reg_strength > 0.:
            self.weight_regularizer = BinaryRegularizer(
                self.weight_reg_strength)
        else:
            self.weight_regularizer = None

        if self.activity_reg_strength > 0.:
            self.activity_regularizer = BinaryActivityRegularization(
                self.activity_reg_strength)
        else:
            self.activity_regularizer = None

        # Bias unnecessary with batch norm.
        self.use_bias = False

        # Batch norm scale.
        self.scale = True

        ####### Network Architecture #######
        self.filters_1 = 128 // shrink
        self.repeats_1 = 2

        self.filters_2 = 256 // shrink
        self.repeats_2 = 2

        self.filters_3 = 512 // shrink
        self.repeats_3 = 2

        self.dense_1 = 1024 // shrink
        self.dense_2 = 1024 // shrink

        self.pool_size = 2
        self.pool_stride = 2
        self.padding = "same"
        ####################################

        # Just for naming layers.
        self.module_id = 0

    def _activation(self, x, name, is_dense=False):
        activation_name = name + "_" + self.activation
        if (self.activation == "binary"):
            if self.activity_regularizer is not None:
                x = self.activity_regularizer(x)
            x = Binarization(name=activation_name)(x)
        else:
            x = Activation(self.activation, name=activation_name)(x)
        return x

    def _dense_block(self, x, units, name, activate=True):
        if self.weight_type == "float":
            x = Dense(units, use_bias=self.use_bias, name=name + "_dense")(x)
        elif self.weight_type == "binary":
            x = BinaryDense(units, name=name + "_dense")(x)
        x = self._batch_norm(x, name=name)
        if activate:
            x = self._activation(x, name=name)
        return x

    def _conv_block(self, x, filters, pool, name, activate=True):
        if self.weight_type == "float":
            layer_name = name
            x = Conv2D(
                filters,
                kernel_size=3,
                use_bias=self.use_bias,
                padding=self.padding,
                name=layer_name + "_conv")(x)
        elif self.weight_type == "binary":
            layer_name = name
            x = BinaryConv2D(
                filters,
                kernel_size=3,
                kernel_regularizer=self.weight_regularizer,
                padding=self.padding,
                name=layer_name + "_conv")(x)
        if pool:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                name=name + "_maxpool")(x)
        x = self._batch_norm(x, name=layer_name)
        if activate:
            x = self._activation(x, name=layer_name)
        return x

    def _batch_norm(self, x, name):
        return BatchNormalization(scale=self.scale, name=name + "_bn")(x)

    def _module(self, x, filters, repeats):
        self.module_id += 1
        name = "m" + str(self.module_id)

        for n in range(repeats):
            block_name = name + "_b" + str(n)
            pool = n == (repeats - 1)
            x = self._conv_block(
                x, filters=filters, pool=pool, name=block_name)
        return x

    def build(self, x):
        x = self._module(x, filters=self.filters_1, repeats=self.repeats_1)
        x = self._module(x, filters=self.filters_2, repeats=self.repeats_2)
        x = self._module(x, filters=self.filters_3, repeats=self.repeats_3)

        if self.dropout_rate > 0.0:
            x = Dropout(rate=self.dropout_rate, name="dropout")(x)

        x = Flatten(name="flatten")(x)

        x = self._dense_block(x, units=self.dense_1, name="fc1")
        x = self._dense_block(x, units=self.dense_2, name="fc2")
        x = self._dense_block(x, self.classes, name="output", activate=False)
        x = Softmax(name="softmax")(x)
        return x
