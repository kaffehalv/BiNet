import numpy as np
import keras as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.datasets import cifar10, cifar100
from keras import optimizers
from keras import callbacks
from SataNet import SataNet
from SimpleNet import SimpleNet
from math import cos, pi

gpus = 1
batch_size = max(128 * gpus, 32)
epochs_half_period = 15
epochs_end = max(epochs_half_period // 5, 2)
epochs = 2 * epochs_half_period + epochs_end
verbose = 2
lr_max = 1.0
lr_init = 1e-1 * lr_max
lr_min = 1e-3 * lr_max
momentum = 0.9
weights_path = "C:/Users/niclas/ML-Projects/cifar/weights.hdf5"


def interpolate(val0, val1, t):
    return val0 + (val1 - val0) * t


def schedule(epoch, lr):
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


optimizer = optimizers.SGD(lr=lr_init, momentum=momentum)
lr_schedule = LearningRateScheduler(schedule, verbose=1)
checkpointer = ModelCheckpoint(
    filepath=weights_path,
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    save_weights_only=True)
callbacks = [lr_schedule, checkpointer]

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

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
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=15,
    width_shift_range=0.01,
    height_shift_range=0.01,
    horizontal_flip=True)
train_datagen.fit(x_train)

val_datagen = ImageDataGenerator(
    samplewise_center=True, samplewise_std_normalization=True)
val_datagen.fit(x_test)

if gpus == 1:
    device = "/gpu:0"
else:
    device = "/cpu:0"

# build the model
with tf.device(device):
    model_input = Input(shape=(32, 32, 3))
    #network = SataNet(32, 32, 3, num_classes)
    network = SimpleNet(32, 32, 3, num_classes)
    model_output = network.build(model_input)
    model = Model(inputs=model_input, outputs=model_output)
    model.summary()
    #model.load_weights(weights_path)

batches_per_epoch = len(x_train) // batch_size

print("Starting training!")
if gpus > 1:
    parallel_model = multi_gpu_model(model, gpus=gpus)

    # Compile model
    parallel_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])

    # Fit the model
    parallel_model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=batches_per_epoch,
        epochs=epochs,
        validation_data=val_datagen.flow(
            x_test, y_test, batch_size=batch_size),
        validation_steps=validation_steps,
        verbose=verbose,
        callbacks=callbacks)
else:
    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])

    # Fit the model
    model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=batches_per_epoch,
        epochs=epochs,
        validation_data=val_datagen.flow(
            x_test, y_test, batch_size=batch_size),
        validation_steps=validation_steps,
        verbose=verbose,
        callbacks=callbacks)

model.load_weights(weights_path)
scores = model.evaluate_generator(
    val_datagen.flow(x_test, y_test, batch_size=batch_size),
    steps=validation_steps)
print("BEST WEIGHTS: loss = %f, acc = %f " % (scores[0], scores[1]))
