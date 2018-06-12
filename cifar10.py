import numpy as np
import keras as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import cifar10, cifar100
from keras import optimizers
from keras import callbacks
from binet import BiNet
from resnet import ResNet
from densenet import DenseNet
from math import log, exp

on_linux = False
gpus = 2
batch_size = max(256 * gpus, 32)
verbose = 2
epochs = 100

optimizer = "sgd"
weight_type = "float"
activation = "relu"
trainable_weights = True
shrink_factor = 1.0
weight_reg_strength = 0.0
activity_reg_strength = 0.0
dropout_rate = 0.0
load_weights = False

# Data augmentation
rotation_range = 0.
width_shift_range = 0.
height_shift_range = 0.
horizontal_flip = True

if optimizer == "adam":
    lr_init = 1e-3
    lr_min = 5e-7
    lr_factor = 0.5
    cooldown = 0
    patience = 5
elif optimizer == "sgd":
    lr_max = 2
    lr_init = 1e-1 * lr_max
    lr_min = 1e-4 * lr_max
    momentum = 0.9
    nesterov = True
    epochs_half_period = (2 * epochs) // 5
    epochs_end = epochs - 2 * epochs_half_period

if on_linux:
    path = "/home/niclasw/BiNet/"
else:
    path = "C:/Users/niclas/ML-Projects/BiNet/"
weights_path = path + "weights.hdf5"
multi_weights_path = path + "multi_weights.hdf5"
best_weights_path = path + "best_weights.hdf5"


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


if optimizer == "adam":
    optimizer = optimizers.Adam(lr=lr_init)
    lr_callback = ReduceLROnPlateau(
        factor=lr_factor,
        cooldown=cooldown,
        patience=patience,
        min_lr=lr_min,
        verbose=1)
elif optimizer == "sgd":
    optimizer = optimizers.SGD(
        lr=lr_init, momentum=momentum, nesterov=nesterov)
    lr_callback = LearningRateScheduler(sgd_schedule, verbose=1)

if gpus > 1:
    checkpoint_weights_path = multi_weights_path
else:
    checkpoint_weights_path = weights_path
checkpointer = ModelCheckpoint(
    filepath=checkpoint_weights_path,
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    save_weights_only=True)
callbacks = [lr_callback, checkpointer]

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_train, y_train), (h, y_test) = cifar100.load_data(label_mode="fine")

# one hot encode outputs
y_train = K.utils.to_categorical(y_train)
y_test = K.utils.to_categorical(y_test)
val_size = y_test.shape[0]
num_classes = y_test.shape[1]
validation_steps = val_size // batch_size

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    horizontal_flip=horizontal_flip)
train_datagen.fit(x_train)

val_datagen = ImageDataGenerator(
    featurewise_center=True, featurewise_std_normalization=True)
val_datagen.fit(x_test)

if gpus == 1:
    device = "/gpu:0"
else:
    device = "/cpu:0"

# build the model
with tf.device(device):
    input_shape = (32, 32, 3)
    model_input = Input(shape=input_shape)
    network = DenseNet(
        weight_type=weight_type,
        activation=activation,
        shrink_factor=shrink_factor,
        weight_reg_strength=weight_reg_strength,
        activity_reg_strength=activity_reg_strength,
        dropout_rate=dropout_rate,
        trainable_weights=trainable_weights,
        input_shape=input_shape,
        classes=num_classes)
    model_output = network.build(model_input)
    model = Model(inputs=model_input, outputs=model_output)
    model.summary()
    if load_weights:
        model.load_weights(best_weights_path)

batches_per_epoch = len(x_train) // batch_size

print("Starting training!")
if gpus > 1:
    parallel_model = multi_gpu_model(model, gpus=gpus)

    # Compile parallel model.
    parallel_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])

    # Fit the model.
    parallel_model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=batches_per_epoch,
        epochs=epochs,
        validation_data=val_datagen.flow(
            x_test, y_test, batch_size=batch_size),
        validation_steps=validation_steps,
        verbose=verbose,
        callbacks=callbacks)

    # Save non-parallel model weights.
    parallel_model.load_weights(multi_weights_path)
    model.save_weights(weights_path)

    # Compile non-parallel model.
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])
else:
    # Compile model.
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])

    # Fit the model.
    model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=batches_per_epoch,
        epochs=epochs,
        validation_data=val_datagen.flow(
            x_test, y_test, batch_size=batch_size),
        validation_steps=validation_steps,
        verbose=verbose,
        callbacks=callbacks)
"""
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
    if "conv" in l.name:
        w = l.get_weights()
        l.set_weights(np.sign(w))

# Test the best (binary) weights.
scores_binary = model.evaluate_generator(
    val_datagen.flow(x_test, y_test, batch_size=batch_size),
    steps=validation_steps)
print("BINARY WEIGHTS: loss = %f, acc = %f " % (scores_binary[0],
                                                scores_binary[1]))
"""
