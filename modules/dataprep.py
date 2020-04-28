#!/usr/bin/env python
# Prepaere the dataset for CNNoperations with Keras.
# This is intentionally seperate from the rest of the dataset_tools
# as it pertains to modifying an existing dataset to use with Keras API
from keras.preprocessing.image import ImageDataGenerator
 #
 # Just a layout as a mental note:
 # Also maybe rename  after labeling?
 # data/
 #     train/
 #         positive/
 #             crack_1.jpg
 #             crack_3.jpg
 #             ...
 #         negative/
 #             crack_1.jpg
 #             crack_3.jpg
 #             ...
 #     test/
 #         positive/
 #             crack_4jpg
 #             crack_2.jpg
 #             ...
 #         negative/
 #             crack_4jpg
 #             crack_2.jpg
 #
 #            ...featurewise_center=False,
    #  samplewise_center=False,
    #  featurewise_std_normalization=False,
    #  samplewise_std_normalization=False,
    #  zca_whitening=False,
    #  rotation_range=0.,
    #  width_shift_range=0.,
    #  height_shift_range=0.,
    #  shear_range=0.,
    #  zoom_range=0.,
    #  channel_shift_range=0.,
    #  fill_mode='nearest',
    #  cval=0.,
    #  horizontal_flip=False,
    #  vertical_flip=False,python static l
    #  rescale=None,
    #  preprocessing_function=None,
    #  data_format=K.image_data_format()

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

#save_to_dir='preview', save_prefix='cat', save_format='jpeg'

# this is the augmentation configuration we will use for testing:
# only rescaling




def MakeBinaryTrainGen(gen,dir, d,b):
 return gen.flow_from_directory(
    dir,
    target_size=(d,d),
    batch_size=b,
    class_mode='binary')

def FitGen(model,)
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
