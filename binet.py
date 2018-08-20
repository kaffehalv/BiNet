from keras.layers import Conv2D, Dense, MaxPooling2D
from keras.layers import BatchNormalization, Flatten, Dropout
from keras.layers import Activation
from keras import backend as K
from quantization_utils import QuantizedConv2D, QuantizedDense
from quantization_utils import Quantization, QuantizationRegularizer


class BiNet():
    def __init__(self,
                 weight_type="quant",
                 activation="quant",
                 weight_bits=1,
                 activation_bits=1,
                 shrink_factor=1.0,
                 weight_reg_strength=0.0,
                 activity_reg_strength=0.0,
                 dropout_rate=0.0,
                 input_shape=(32, 32, 3),
                 classes=10):
        self.weight_type = weight_type
        self.activation = activation
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_reg_strength = weight_reg_strength
        self.activity_reg_strength = activity_reg_strength
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.classes = classes

        # Set up the regularizers.
        if self.weight_reg_strength > 0.0:
            self.weight_regularizer = QuantizationRegularizer(
                scale=self.weight_reg_strength)
        else:
            self.weight_regularizer = None

        if self.activity_reg_strength > 0.0:
            self.activity_regularizer = QuantizedActivityRegularization(
                scale=self.activity_reg_strength)
        else:
            self.activity_regularizer = None

        self.weight_initializer = "glorot_uniform"

        # Bias unnecessary with batch norm.
        self.use_bias = False

        # Batch norm scale.
        self.scale = True

        ####### Network Architecture #######
        self.filters_1 = int(shrink_factor * 128)
        self.repeats_1 = 2

        self.filters_2 = int(shrink_factor * 256)
        self.repeats_2 = 2

        self.filters_3 = int(shrink_factor * 512)
        self.repeats_3 = 2

        self.dense_1 = int(shrink_factor * 1024)
        self.dense_2 = int(shrink_factor * 1024)

        self.pool_size = 2
        self.pool_stride = 2
        self.padding = "same"
        ####################################

        # Just for naming layers.
        self.module_id = 0

    def _activation(self, x, activation, name, is_dense=False):
        activation_name = name + "_" + activation
        if (activation == "quant"):
            x = Quantization(
                name=activation_name, num_bits=self.activation_bits)(x)
        elif (activation == "clip"):
            x = Lambda(lambda x: K.clip(x, -1., 1.), name=activation_name)(x)
        else:
            x = Activation(activation, name=activation_name)(x)
            if self.activity_regularizer is not None:
                x = self.activity_regularizer(x)
        return x

    def _batch_norm(self, x, name):
        return BatchNormalization(scale=self.scale, name=name + "_bn")(x)

    def _dense_block(self, x, units, activation, name):
        if len(x.shape) > 2:
            x = Flatten(name="flatten")(x)

        if (self.weight_type == "float"):
            x = Dense(
                units,
                use_bias=self.use_bias,
                kernel_initializer=self.weight_initializer,
                kernel_regularizer=self.weight_regularizer,
                name=name + "_dense")(x)
        elif (self.weight_type == "quant"):
            x = QuantizedDense(
                units,
                use_bias=self.use_bias,
                kernel_initializer=self.weight_initializer,
                num_bits=self.weight_bits,
                name=name + "_dense")(x)

        x = self._batch_norm(x, name=name)
        x = self._activation(x, activation=activation, name=name)
        return x

    def _conv_unit(self,
                   x,
                   name,
                   filters,
                   activation,
                   kernel_size=3,
                   downsample=False,
                   separable=False):
        if self.weight_type == "float":
            x = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                use_bias=self.use_bias,
                kernel_initializer=self.weight_initializer,
                kernel_regularizer=self.weight_regularizer,
                padding=self.padding,
                name=name + "_conv")(x)
        elif (self.weight_type == "quant"):
            x = QuantizedConv2D(
                filters=filters,
                kernel_size=kernel_size,
                use_bias=self.use_bias,
                kernel_initializer=self.weight_initializer,
                kernel_regularizer=self.weight_regularizer,
                padding=self.padding,
                num_bits=self.weight_bits,
                name=name + "_conv")(x)

        if downsample:
            x = MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding,
                name=name + "_pool")(x)

        x = self._batch_norm(x, name=name)
        x = self._activation(x, activation=activation, name=name)
        return x

    def _module(self, x, filters, repeats, activation):
        name = "m" + str(self.module_id)

        for n in range(repeats):
            block_name = name + "_b" + str(n)
            downsample = (n == repeats - 1)
            x = self._conv_unit(
                x,
                filters=filters,
                downsample=downsample,
                activation=activation,
                name=block_name)

        self.module_id += 1
        return x

    def build(self, x):
        x = self._module(
            x,
            filters=self.filters_1,
            repeats=self.repeats_1,
            activation=self.activation)
        x = self._module(
            x,
            filters=self.filters_2,
            repeats=self.repeats_2,
            activation=self.activation)
        x = self._module(
            x,
            filters=self.filters_3,
            repeats=self.repeats_3,
            activation=self.activation)

        if self.dropout_rate > 0.0:
            x = Dropout(rate=self.dropout_rate, name="dropout")(x)

        x = self._dense_block(
            x, units=self.dense_1, activation=self.activation, name="dense_1")
        x = self._dense_block(
            x, units=self.dense_2, activation=self.activation, name="dense_2")
        x = self._dense_block(
            x, units=self.classes, activation="softmax", name="output")
        return x
