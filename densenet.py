from keras.layers import Conv2D, SeparableConv2D, Dense, AveragePooling2D
from keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.layers import PReLU, Activation, Lambda
from keras.layers import Concatenate
from keras import backend as K
from quantization_utils import QuantizedConv2D, QuantizedDense, QuantizedDepthwiseConv2D
from quantization_utils import Quantization, QuantizedActivityRegularization, QuantizationRegularizer


class DenseNet():
    def __init__(self,
                 weight_type="quant",
                 activation="quant",
                 weight_bits=8,
                 activation_bits=8,
                 separable=False,
                 shrink_factor=1.0,
                 weight_reg_strength=0.0,
                 activity_reg_strength=0.0,
                 dropout_rate=0.0,
                 trainable_weights=True,
                 input_shape=(32, 32, 3),
                 classes=10):
        self.weight_type = weight_type
        self.activation = activation
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.separable = separable
        self.weight_reg_strength = weight_reg_strength
        self.activity_reg_strength = activity_reg_strength
        self.dropout_rate = dropout_rate
        self.trainable_weights = trainable_weights
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

        # If to scale binary approximations by mean(abs(W)) channelwise.
        # See the XNOR-Net paper.
        self.scale_kernel = False

        ####### Network Architecture #######
        self.filters_0 = int(shrink_factor * 16)

        self.filters_1 = int(shrink_factor * 16)
        self.repeats_1 = 3
        self.downsample_1 = False

        self.filters_2 = int(shrink_factor * 16)
        self.repeats_2 = 4
        self.downsample_2 = True

        self.filters_3 = int(shrink_factor * 16)
        self.repeats_3 = 4
        self.downsample_3 = True

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
        elif (activation == "silu"):
            x = Lambda(lambda x: x * K.sigmoid(x), name=activation_name)(x)
        elif (activation == "prelu"):
            x = PReLU(shared_axes=[1, 2, 3], name=activation_name)(x)
        else:
            x = Activation(activation, name=activation_name)(x)
            if self.activity_regularizer is not None:
                x = self.activity_regularizer(x)
        return x

    def _batch_norm(self, x, name):
        return BatchNormalization(scale=self.scale, name=name + "_bn")(x)

    def _dense_block(self, x, units, activation, name, float_override=False):
        x = self._batch_norm(x, name=name)
        if float_override:
            x = self._activation(x, activation="relu", name=name)
        else:
            x = self._activation(x, activation=activation, name=name)

        if len(x.shape) > 2:
            x = Flatten(name="flatten")(x)

        if (self.weight_type == "float") or float_override:
            x = Dense(
                units,
                use_bias=self.use_bias,
                kernel_initializer=self.weight_initializer,
                kernel_regularizer=self.weight_regularizer,
                name=name + "_dense")(x)
        elif (self.weight_type == "quant"):
            x = QuantizedDense(
                units,
                scale_kernel=self.scale_kernel,
                kernel_initializer=self.weight_initializer,
                trainable=self.trainable_weights,
                num_bits=self.weight_bits,
                name=name + "_dense")(x)
        return x

    def _conv_unit(self,
                   x,
                   name,
                   filters,
                   activation,
                   kernel_size=3,
                   stride=1,
                   first_layer=False,
                   dropout_rate=0.0,
                   separable=False):
        if not first_layer:
            x = self._batch_norm(x, name=name)
            x = self._activation(x, activation=activation, name=name)

        if self.weight_type == "float":
            if separable:
                x = SeparableConv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    use_bias=self.use_bias,
                    depthwise_initializer=self.weight_initializer,
                    depthwise_regularizer=self.weight_regularizer,
                    pointwise_initializer=self.weight_initializer,
                    pointwise_regularizer=self.weight_regularizer,
                    padding=self.padding,
                    name=name + "_conv")(x)
            else:
                x = Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    use_bias=self.use_bias,
                    kernel_initializer=self.weight_initializer,
                    kernel_regularizer=self.weight_regularizer,
                    padding=self.padding,
                    name=name + "_conv")(x)
        elif (self.weight_type == "quant"):
            if separable:
                x = QuantizedDepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=stride,
                    depthwise_initializer=self.weight_initializer,
                    depthwise_regularizer=self.weight_regularizer,
                    padding=self.padding,
                    trainable=self.trainable_weights,
                    num_bits=self.weight_bits,
                    name=name + "_dw")(x)
                x = self._batch_norm(x, name=name + "_mid")
                x = self._activation(
                    x, activation=activation, name=name + "_mid")
                x = QuantizedConv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    scale_kernel=self.scale_kernel,
                    kernel_initializer=self.weight_initializer,
                    kernel_regularizer=self.weight_regularizer,
                    padding=self.padding,
                    trainable=self.trainable_weights,
                    num_bits=self.weight_bits,
                    name=name + "_pw")(x)
            else:
                x = QuantizedConv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    scale_kernel=self.scale_kernel,
                    kernel_initializer=self.weight_initializer,
                    kernel_regularizer=self.weight_regularizer,
                    padding=self.padding,
                    trainable=self.trainable_weights,
                    num_bits=self.weight_bits,
                    name=name + "_conv")(x)

        if dropout_rate > 0.0 and not first_layer:
            x = Dropout(rate=dropout_rate, name=name + "_dropout")(x)
        return x

    def _res_block(self,
                   x_in,
                   name,
                   filters,
                   activation,
                   kernel_size=3,
                   downsample=False,
                   dropout_rate=0.0):
        if downsample:
            stride = 2
            x_pool = AveragePooling2D(
                pool_size=kernel_size,
                strides=stride,
                padding=self.padding,
                name=name + "_pool")(x_in)
        else:
            stride = 1
            x_pool = x_in

        x_conv = self._conv_unit(
            x_in,
            filters=filters,
            stride=stride,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            activation=activation,
            separable=self.separable,
            name=name)

        x = Concatenate(name=name + "_concat")([x_pool, x_conv])
        return x

    def _module(self, x, filters, repeats, downsample, activation):
        name = "m" + str(self.module_id)

        for n in range(repeats):
            block_name = name + "_b" + str(n)
            x = self._res_block(
                x,
                filters=filters,
                downsample=(downsample and n == 0),
                dropout_rate=self.dropout_rate,
                activation=activation,
                name=block_name)

        self.module_id += 1
        return x

    def build(self, x):
        x = self._conv_unit(
            x,
            filters=self.filters_0,
            activation=self.activation,
            first_layer=True,
            name="first")

        x = self._module(
            x,
            filters=self.filters_1,
            repeats=self.repeats_1,
            downsample=self.downsample_1,
            activation=self.activation)
        x = self._module(
            x,
            filters=self.filters_2,
            repeats=self.repeats_2,
            downsample=self.downsample_2,
            activation=self.activation)
        x = self._module(
            x,
            filters=self.filters_3,
            repeats=self.repeats_3,
            downsample=self.downsample_3,
            activation=self.activation)

        x = self._conv_unit(
            x,
            filters=self.classes,
            kernel_size=1,
            activation=self.activation,
            name="last")

        x = GlobalAveragePooling2D(name="global_avg_pool")(x)

        x = self._batch_norm(x, name="output")
        x = self._activation(x, activation="softmax", name="output")
        return x
