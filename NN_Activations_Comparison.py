# @author: BOYU ZHANG
# GitHub: CristianoBY
# All Rights Reserved

# Exploring different performance of various activation functions on Neural Network
# basic activation functions: sigmoid, relu, and tanh
# advanced activation functions: LeakyReLU and ELU

# Test Case: Neural Network for solving the MNIST dataset recognition
# http://yann.lecun.com/exdb/mnist/
# Comparing cases: Sigmoid VS. ReLU VS. tanh, ELU VS. tanh, and ELU VS. LeakyReLU

import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
# I use Adam Optimizer
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU, ELU

# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

m = 10
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, m)
y_test = keras.utils.to_categorical(y_test, m)

print('Train size:', x_train.shape[0])
print('Test size:', x_test.shape[0])

# training parameter for all networks
batch_size = 128
epochs = 25

# Build the network
# try all the basic activations
for activation in ['sigmoid', 'tanh', 'relu']:

    model = Sequential()
    model.add(Dense(512, activation = activation, input_shape = (784,), kernel_initializer = "uniform"))
    # using sofmax for output layer
    model.add(Dense(m, activation = 'softmax'))

    print("\nNeural Network with Activation: ", activation)

    # layer basic info and size
    for l in model.layers:
        print(l.name, l.input_shape,'==>',l.output_shape)
    print(model.summary())

    model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(),
              metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (x_test, y_test))
    
    plt.plot(history.history['val_acc'])

    
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['sigmoid', 'tanh', 'relu'], loc = 'upper right')
plt.show()

# try advanced activation layers
for adv_layer in [ LeakyReLU(alpha = 0.3), ELU(alpha = 1.0)]:

    model = Sequential()
    model.add(Dense(512, input_shape = (784,), kernel_initializer = "uniform"))

    # add the advanced layer as a class
    model.add(adv_layer)
    model.add(Dense(m, activation = 'softmax'))

    print("\nNeural Network with Advanced Activation: ", adv_layer)

    # layer basic info and size
    for l in model.layers:
        print(l.name, l.input_shape,'==>',l.output_shape)
    print(model.summary())

    model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(),
              metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (x_test, y_test))
    plt.plot(history.history['val_acc'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['LeakyReLU', 'ELU'], loc = 'upper right')
plt.show()


