# @author: BOYU ZHANG
# GitHub: CristianoBY
# All Rights Reserved

# Neural Network for solving the MNIST dataset recognition
# http://yann.lecun.com/exdb/mnist/

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
print('Test size:', x_test.shape[0], '\n')

# build the neural network
# simple two layer NN with sigmoid activation at 1st layer and softmax at output layer
model = Sequential()
model.add(Dense(512, input_shape = (784,), activation = 'sigmoid', kernel_initializer = "uniform"))
model.add(Dense(m, activation = 'softmax'))

print('Simple 2-layer NN with sigmoid activation at 1st layer and softmax at output layer')
for l in model.layers:
    print (l.name, l.input_shape,'==>',l.output_shape)

print ('\n', model.summary())

# training parameter for all networks
batch_size = 128
epochs = 5

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
    
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=100)

print ('Test loss:', round(score[0], 3))
print ('Test accuracy:', round(score[1], 3))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

