import numpy
import keras as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10, cifar100
from keras import optimizers
from keras import callbacks
from SataNet import SataNet

gpus = 2
batch_size = max(128 * gpus, 32)
epochs = 100
verbose = 2
lr_init = 0.01
lr_patience = 10
lr_factor = 0.5
min_lr = 1e-4
optimizer = optimizers.Adam(lr=lr_init)

# fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

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
    rotation_range=30,
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
    network = SataNet(32, 32, 3, num_classes)
    model_output = network.build(model_input)
    model = Model(inputs=model_input, outputs=model_output)
    model.summary()

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=min_lr)

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
        steps_per_epoch=len(x_train) / batch_size,
        epochs=epochs,
        validation_data=val_datagen.flow(
            x_test, y_test, batch_size=batch_size),
        validation_steps=validation_steps,
        verbose=verbose,
        callbacks=[reduce_lr])
else:
    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])

    # Fit the model
    model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) / batch_size,
        epochs=epochs,
        validation_data=val_datagen.flow(
            x_test, y_test, batch_size=batch_size),
        validation_steps=validation_steps,
        verbose=verbose,
        callbacks=[reduce_lr])
