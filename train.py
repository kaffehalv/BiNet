import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import cifar10, cifar100
from keras import optimizers
from keras import callbacks
from binet import BiNet
from math import log, exp

batch_size = 256
verbose = 2
epochs = 10

optimizer = "adam"
weight_type = "quant"
activation = "quant"
weight_bits = 1
activation_bits = 1
shrink_factor = 1.0
weight_reg_strength = 1e-3
dropout_rate = 0.0
load_weights = False

# Data augmentation
rotation_range = 0
width_shift_range = 0.0
height_shift_range = 0.0
horizontal_flip = True

if optimizer == "adam":
    lr_init = 1e-3
    lr_min = 5e-7
    lr_factor = 0.1
    cooldown = 0
    patience = 10
elif optimizer == "sgd":
    lr_max = 1
    lr_init = 1e-1 * lr_max
    lr_min = 1e-4 * lr_max
    momentum = 0.9
    nesterov = True
    epochs_half_period = (2 * epochs) // 5
    epochs_end = epochs - 2 * epochs_half_period

weights_path = "weights.hdf5"


def interpolate(val0, val1, t):
    return exp(log(val0) + (log(val1) - log(val0)) * t)


def sgd_schedule(epoch, lr):
    # One-cycle learning rate schedule.
    t0 = epoch % epochs_half_period
    if epoch < epochs_half_period:
        t = t0 / (epochs_half_period - 1)
        new_lr = interpolate(lr_init, lr_max, t)
    elif epoch < 2 * epochs_half_period:
        t = t0 / (epochs_half_period - 1)
        new_lr = interpolate(lr_max, lr_init, t)
    else:
        t = t0 / (epochs_end - 1)
        new_lr = interpolate(lr_init, lr_min, t)
    return new_lr


checkpointer = ModelCheckpoint(
    filepath=weights_path,
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    save_weights_only=True)

if optimizer == "adam":
    optimizer = optimizers.Adam(lr=lr_init)
    lr_reduce = ReduceLROnPlateau(
        monitor="val_acc",
        factor=lr_factor,
        cooldown=cooldown,
        patience=patience,
        min_lr=lr_min,
        verbose=1)
    callbacks = [lr_reduce, checkpointer]
elif optimizer == "sgd":
    optimizer = optimizers.SGD(
        lr=lr_init, momentum=momentum, nesterov=nesterov)
    lr_schedule = LearningRateScheduler(sgd_schedule, verbose=1)
    callbacks = [lr_schedule, checkpointer]

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_train, y_train), (h, y_test) = cifar100.load_data(label_mode="fine")

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
val_size = y_test.shape[0]
num_classes = y_test.shape[1]
validation_steps = val_size // batch_size


def preprocess(x):
    return 2 / 255 * x - 1


train_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    horizontal_flip=horizontal_flip,
    preprocessing_function=preprocess)
train_datagen.fit(x_train)

val_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    preprocessing_function=preprocess)
val_datagen.fit(x_test)

input_shape = (32, 32, 3)
model_input = Input(shape=input_shape)
network = BiNet(
    weight_type=weight_type,
    activation=activation,
    weight_bits=weight_bits,
    activation_bits=activation_bits,
    shrink_factor=shrink_factor,
    weight_reg_strength=weight_reg_strength,
    dropout_rate=dropout_rate,
    input_shape=input_shape,
    classes=num_classes)
model_output = network.build(model_input)
model = Model(inputs=model_input, outputs=model_output)
model.summary()
if load_weights:
    model.load_weights(weights_path)

batches_per_epoch = len(x_train) // batch_size

print("Starting training!")
# Compile model.
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Fit the model.
model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=batches_per_epoch,
    epochs=epochs,
    validation_data=val_datagen.flow(x_test, y_test, batch_size=batch_size),
    validation_steps=validation_steps,
    verbose=verbose,
    callbacks=callbacks)

# Test the best (float) weights.
model.load_weights(weights_path)
scores = model.evaluate_generator(
    val_datagen.flow(x_test, y_test, batch_size=batch_size),
    steps=validation_steps)
print("FLOAT WEIGHTS: loss = %f, acc = %f " % (scores[0], scores[1]))

# Print some statistics.
for l in model.layers:
    if "conv" in l.name:
        w = l.get_weights()[0]
        mu = np.mean(w)
        sigma = np.sqrt(np.mean((w - mu)**2))
        minval = np.amin(w)
        maxval = np.amax(w)
        print("mu = %f, sigma = %f, min = %f, max = %f" % (mu, sigma, minval,
                                                           maxval))
# Binarize all weights.
for l in model.layers:
    if "conv" in l.name or "dense" in l.name:
        w = l.get_weights()
        l.set_weights(np.sign(w))

# Test the best (binary) weights.
scores_binary = model.evaluate_generator(
    val_datagen.flow(x_test, y_test, batch_size=batch_size),
    steps=validation_steps)
print("BINARY WEIGHTS: loss = %f, acc = %f " % (scores_binary[0],
                                                scores_binary[1]))
