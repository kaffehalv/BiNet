from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import Flatten, Dropout, MaxPooling2D, Dense
from keras import backend as K
from keras.initializers import RandomUniform, TruncatedNormal
from binary_utils import BinaryConv2D, BinaryDense, Binarization
from binary_utils import BinaryActivityRegularization, BinaryRegularizer


class BiNet():
    def __init__(self,
                 weight_type="binary",
                 activation="binary",
                 shrink_factor=1.0,
                 weight_reg_strength=0.0,
                 activity_reg_strength=0.0,
                 dropout_rate=0.0,
                 trainable_weights=True,
                 input_shape=(32, 32, 3),
                 classes=10):
        self.weight_type = weight_type
        self.activation = activation
        self.weight_reg_strength = weight_reg_strength
        self.activity_reg_strength = activity_reg_strength
        self.dropout_rate = dropout_rate
        self.trainable_weights = trainable_weights
        self.input_shape = input_shape
        self.classes = classes

        use_range = True

        # Set up the regularizers.
        if self.weight_reg_strength > 0.:
            self.weight_regularizer = BinaryRegularizer(
                scale=self.weight_reg_strength, use_range=use_range)
        else:
            self.weight_regularizer = None

        if self.activity_reg_strength > 0.:
            self.activity_regularizer = BinaryActivityRegularization(
                scale=self.activity_reg_strength, use_range=use_range)
        else:
            self.activity_regularizer = None

        if self.weight_type == "float":
            self.weight_initializer = "glorot_uniform"
        else:
            init_lim = 0.5
            self.weight_initializer = TruncatedNormal(
                mean=0., stddev=0.5 * init_lim)

        # Bias unnecessary with batch norm.
        self.use_bias = False

        # Batch norm scale.
        self.scale = True

        # If to scale binary approximations by mean(abs(W)) channelwise.
        # See the XNOR-Net paper.
        self.scale_kernel = False

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
        if (activation == "binary"):
            x = Binarization(name=activation_name)(x)
        elif (activation == "clip"):
            x = Lambda(lambda x: K.clip(x, -1., 1.), name=activation_name)(x)
        elif (activation == "silu"):
            x = Lambda(lambda x: x * K.sigmoid(x), name=activation_name)(x)
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
        elif self.weight_type == "binary":
            x = BinaryDense(
                units,
                scale_kernel=self.scale_kernel,
                kernel_initializer=self.weight_initializer,
                trainable=self.trainable_weights,
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
                   downsample=False):
        if (self.weight_type == "float"):
            x = Conv2D(
                filters,
                kernel_size=kernel_size,
                use_bias=self.use_bias,
                kernel_initializer=self.weight_initializer,
                kernel_regularizer=self.weight_regularizer,
                padding=self.padding,
                name=name + "_conv")(x)
        elif self.weight_type == "binary":
            x = BinaryConv2D(
                filters,
                kernel_size=kernel_size,
                scale_kernel=self.scale_kernel,
                kernel_initializer=self.weight_initializer,
                kernel_regularizer=self.weight_regularizer,
                padding=self.padding,
                trainable=self.trainable_weights,
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
