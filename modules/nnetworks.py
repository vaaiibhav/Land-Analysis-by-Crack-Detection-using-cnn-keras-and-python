#!/usr/bin/env python
import keras.layers
import sys

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import *
from keras.constraints import maxnorm

from keras.optimizers import SGD
from keras.metrics import binary_accuracy

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

key={'nc':'number of classes',
     'b':'batch_size',
     'h':'input height/input_shape[0]',
     'w':'input width/input_shape[1]',
     'k':'kernel_size',
     'fn':'activation(s)',
     }
def help():
    print(key)
# keras.layers.convolutional.Conv2D(filters, kernel_size,
#                                   strides=(1, 1), padding='valid',
#                                   data_format=None, dilation_rate=(1, 1),
#                                   activation=None, use_bias=True,
#                                   kernel_initializer='glorot_uniform',
#                                   bias_initializer='zeros', kernel_regularizer=None,
#                                   bias_regularizer=None, activity_regularizer=None,
#                                   kernel_constraint=None, bias_constraint=None)

#def flex_get(): need this for a network constructor of simple and complex designs

#LayerTypes{'I','C','Cr','Do','M','F','D','O'}
# A pythonic way would be actuall kv pairs with references to layer constructors

#Lazy mode on: please start with I and end with O.
# I will implement checking first/last iter later.
#also has the potetial to be a @classmethod
def modelmaker(name,nc, layers,b, sz,k=3,a_fn='relu',dr=.3, mp=2,den=200,fc_scale=2,cr=2):
  model = Sequential()
  for ly in layers:
    if(ly=='I'):
        model.add(Conv2D(b, (k, k), input_shape=(sz, sz, 3),
         padding='same', activation=a_fn, kernel_constraint=maxnorm(3)))

    elif(ly=='C'):
        model.add(Conv2D(b, (k, k), activation=a_fn, padding='same', kernel_constraint=maxnorm(3)))
    elif(ly=='Do'):
        model.add(Dropout(dr))
    elif(ly=='D'):
        model.add(Dense(den, activation=fn[2], kernel_constraint=maxnorm(3)))
        den//=fc_scale
    elif(ly=='M'):
        model.add(MaxPooling2D(pool_size=(mp, mp)))
        #sz//=mp
    elif(ly=='Cr'):
        model.add(Cropping2D(cropping=((cr, cr), (cr, cr))))
    elif(ly=='F'):
        model.add(Flatten())

    elif(ly=='O'):
        model.add(Dense(nc, activation='softmax'))

  saveModelJSON(model, 'dr_nn1')
  return model


@staticmethod
def cnn2():
    return modelmaker('cnn2',['I','C','M','M','Cr','F','D','D','O'],64,64)

def cifar_network(nc,b=100,h=32,w=32, k=3, fn=['relu','relu','relu','softmax']):
    model = Sequential()
    model.add(Conv2D(b, (k, k), input_shape=(h, w, 3), padding='same', activation=fn[0], kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(b, (k, k), activation=fn[1], padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation=fn[2], kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(nc, activation=fn[3]))

    return model

# Is there padding in Zhang's network?
# it is losing 4,4 then 3 pixels for each mp->conv
def zhang_network(nc,b=48,h=99,w=99,fn='relu'):
    #add dropouts
    model = Sequential()
    model.add(Conv2D(b, (4, 4), input_shape=(99, 99, 3)))
    model.add(Cropping2D(cropping=((2, 1), (1, 2))))
    model.add(Conv2D(b, (4, 4), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #flatten here?
    model.add(Cropping2D(cropping=((2, 2), (2, 2))))
    model.add(Conv2D(b, (5, 5), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Cropping2D(cropping=((2, 2), (2, 2))))
    model.add(Conv2D(b, (3, 3), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Cropping2D(cropping=((2, 1), (1, 2))))
    model.add(Conv2D(b, (4, 4), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(nc, activation='softmax'))

    return model


#A ModelFactory would be nice
#def dr_network(name,nc=2,b=48,h=99,w=99,fn='relu'):
def dr_network(size=64, b=60, fn='relu', nc=2):
    #add dropouts
    model = Sequential()
    model.add(Conv2D(b, (4, 4), input_shape=(size, size, 3), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #flatten here?
    model.add(Conv2D(b, (5, 5), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(b, (3, 3), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(b, (2, 2), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.33))
    model.add(Flatten())
    model.add(Dense(400, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(nc, activation='softmax'))

    #saveModelJSON(model, 'dr_nn1')
    return model


def saveModelJSON(model, name, path='../models/'):
    jsonfile=open(path+name,w)
    jsonfile.write(model.to_json())

def modelFromData(jsonfile, weights):
    #load json
    model = model_from_json(json_string)
    if weights:
        model.load_weights(file)
    return model


    # li = iter(object_list)
    #
    # obj = next(li)
    #
    # do_first_thing_with(obj)
    #
    # while True:
    #     try:
    #         do_something_with(obj)
    #         obj = next(li)
    #     except StopIteration:
    #         do_final_thing_with(obj)
    #         break
    #
