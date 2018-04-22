# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:09:07 2018

@author: ANKIT
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
 
# Image dimensions
img_width, img_height = 150, 150 
 
"""
    Creates a CNN model
    p: Dropout rate
    input_shape: Shape of input 
"""
def create_model(p, input_shape=(32, 32, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer 
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p/2))
    model.add(Dense(15, activation='sigmoid'))
    
    # Compiling the CNN
    optimizer = Adam(lr=1e-2)
    metrics=['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model
"""
    Fitting the CNN to the images.
"""
def run_training(bs=32, epochs=10):
    
    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                       shear_range = 0.2, 
                                       zoom_range = 0.2, 
                                       horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
 
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (img_width, img_height),
                                                 batch_size = bs,
                                                 class_mode = 'categorical')
                                                 
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_width, img_height),
                                            batch_size = bs,
                                            class_mode = 'categorical')
    
    # Helper - Callbacks
    modelCheckPtr = ModelCheckpoint('/models/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

                                    
    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))                                  
    model.fit_generator(training_set,
                         steps_per_epoch=46691/bs,
                         epochs = epochs,
                         workers=8,
                         validation_data = test_set,
                         validation_steps = 15567/bs,
                         callbacks=[modelCheckPtr, earlyStopping])
def main():
    run_training(bs=32, epochs=10)
 
""" Main """
if __name__ == "__main__":
    main()