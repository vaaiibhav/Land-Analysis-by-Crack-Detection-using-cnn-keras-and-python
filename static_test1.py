#!/usr/bin/env python
# Simple CNN model for CIFAR-10
import numpy
import argparse

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

from modules.nnetworks import *
from results import *
## foundation +plaster

ready_gen = ImageDataGenerator(rescale=1./255,)

prep_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

model= dr_network()

sgd=SGD(0.05,0.9) #ter is also a decay
loss='binary_crossentropy'
metrics=[binary_accuracy]

model.compile(sgd,loss,metrics)

tgt_size=(128,128)
batch_size=42
train_gen = prep_gen.flow_from_directory(
    '../data/walls/train',
    target_size=tgt_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['solid','cracked'])

validation_generator = ready_gen.flow_from_directory(
    '../data/walls/test',
    target_size=tgt_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['solid','cracked'])

#add saving

# train model
log=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
#validation_data=x,y
model.save_weights( savename + '.h5')

# Final evaluation of the model
scores = model.evaluate_generator(validation_generator, steps, max_q_size=10,
                                  workers=1, pickle_safe=False)
(X_test, y_test, 1)
print(scores)
print(model.metrics_names)
saveModelJSON(model,'test')
