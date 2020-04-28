#!/usr/bin/env python3

from nnetworks import *
from keras.optimizers import SGD
from keras.metrics import binary_accuracy

seed = 7

K.set_floatx(ftype)

def trainRoutine(model, data, epoch, batch, savename):

    #Input-> load data
    #(so folders?)
    sgd=SGD(0.05,0.9) #ter is also a decay
    loss='binary_crossentropy'
    metrics=[binary_accuracy]

    model.compile(sgd,loss,metrics)

    # train model
    log=model.fit(X_train, y_train,batch , epochs, validation_split=0.4))
            #validation_data=x,y
    model.save_weights( savename + '.h5')

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, 1)
    print(scores)
    saveModelJSON(model,'test')
    # save plot data/ plot  if u categorical_crossentropy
    #use your results module

    #return model? Do i want a copied instance?
    return  model, log, scores
