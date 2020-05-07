### DESCRIPTION
### This file will train a simple cnn to classify images in the folder data/1x/
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from os.path import isfile, join


# Define a nice plot function for the accuracy and loss over time
# History is the object returns by a model.fit()
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# loads the data given, by the data_path
def load_data(data_path, test_fraction):
    # get all file names
    bananen_onlyfiles   = [f for f in listdir(join(data_path, 'Banana')) if isfile(join(data_path, 'Banana', f))]
    carambola_onlyfiles = [f for f in listdir(join(data_path, 'Carambola')) if isfile(join(data_path, 'Carambola', f))]

    # create data arrays shuffled
    bananen    = np.asarray([np.array(Image.open(join(data_path, 'Banana', banaan))) for banaan in bananen_onlyfiles])
    np.random.shuffle(bananen)
    carambolas = np.asarray([np.array(Image.open(join(data_path, 'Carambola', carambola))) for carambola in carambola_onlyfiles])
    np.random.shuffle(carambolas)

    #x train
    bananen_x_train    = bananen[0:int(1-test_fraction*len(bananen))]
    carambolas_x_train = carambolas[0:int(1-test_fraction*len(carambolas))]
    
    #x test
    bananen_x_test     = bananen[int(1-test_fraction*len(bananen)):len(bananen)]
    carambolas_x_test  = carambolas[int(1-test_fraction*len(carambolas)):len(carambolas)]

    #combine the two classes - normalize
    x_train = np.concatenate([bananen_x_train, carambolas_x_train])/255.0
    x_test  = np.concatenate([bananen_x_test , carambolas_x_test])/255.0

    #combine the label vectors
    y_train = np.concatenate((np.zeros(bananen_x_train.shape[0]), np.ones(carambolas_x_train.shape[0])))
    y_test = np.concatenate((np.zeros(bananen_x_test.shape[0]),   np.ones(carambolas_x_test.shape[0])))

    if VERBOSE:
        print("bananen.shape: " + str(bananen.shape))
        print("carambolas.shape: " + str(carambolas.shape))
        print()

        print("bananen_x_train.shape: " + str(bananen_x_train.shape))
        print("carambolas_x_train.shape: " + str(carambolas_x_train.shape))
        print()

        print("bananen_x_test.shape: " + str(bananen_x_test.shape))
        print("carambolas_x_test.shape: " + str(carambolas_x_test.shape))

    return (x_train, y_train), (x_test, y_test)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    if VERBOSE:
        model.summary()
    return model



################################################################# script part


# OTHER variables
data_path = r'data/1x/'
IMG_DIMS = (64,80,3)
VERBOSE = True
NET_VERBOSE = True
model_path = "models_trained/model_1x.h5"
checkpoint_path = "checkpoints/cp_1x.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)



# NETWORK parameters:
BATCH_SIZE = 4
EPOCHS = 12
TEST_FRACTION = 0.2         #fraction of test images of total images.  0.2 = 20%

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_best_only=True)


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data(data_path, TEST_FRACTION)
print("\n---------------------------------------------\n")
print("x_train.shape" + str(x_train.shape))
print("y_train.shape" + str(y_train.shape))
print("x_test.shape" + str(x_test.shape))
print("y_test.shape" + str(y_test.shape))
print("example image = " + str(x_train[4]))
print("\n---------------------------------------------\n")

model = create_model()

history = model.fit(x_train, y_train,
                   epochs=EPOCHS,
                   verbose=NET_VERBOSE,
                   validation_data=(x_test, y_test),
                   batch_size=BATCH_SIZE,
                   callbacks=[cp_callback])


loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


#plot the accuracy of the model over time/epochs
plot_history(history)