# Various Neural Networks Projects by TensorFlow
This repository contains different kinds of Neural Networks projects implemented in TensorFlow.

# Requirements
------------------- ---------
jupyter             1.0.0

Keras               2.2.0

matplotlib          2.2.2

numpy               1.14.5

pandas              0.23.3

scipy               1.1.0

seaborn             0.8.1

six                 1.11.0

tensorflow          1.9.0

virtualenv          16.0.0

*Notice: this list is NOT complete
------------------- ---------
# Update Thu. July 26, 2018

## Convolutional Neural Network for the Quick Draw Game

>QuickDraw

Convolutional Neural Network for a QuickDraw classifier trained by dataset provided by Google

https://console.cloud.google.com/storage/browser/quickdraw_dataset/?pli=1

I used the simplified data in form of numpy bitmap provided by Google and the Quick Draw AI Experiments

https://quickdraw.withgoogle.com/data

Check

>QD.ipynb

for details

# Update Thu. July 12, 2018

## Neural Network for MNIST Dataset Recognition:

>2-Layer-MNIST.py

Simple 2-layer Neural Network with sigmoid activation for 1st layer and softmax activation for output layer

--The Final Accuracy:

![MNIST-Result](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/MNIST.png)

--Loss VS. Epoches

![Loss VS. Epoches](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/2-Layer_MNIST.png)

## Exploring different performance of various activation functions on Neural Network:

>NN_Activations_Comparison.py

Basic activation functions: sigmoid, relu, and tanh.

Advanced activation functions: LeakyReLU and ELU.

Test Case: Neural Network for solving the MNIST dataset recognition
http://yann.lecun.com/exdb/mnist/

Comparing Results (Accuracy VS. Epoches): 

--Sigmoid VS. ReLU VS. tanh

![Sigmoid VS. ReLU VS. tanh](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/sig-relu-tanh.png)

### In my 2-layer NN with softmax as output activation function, ReLU, as the first layer, is behaving surprisingly bad comparing to sigmoid and tanh. I will keep updating the reason behind. 

### ELU and LeakyReLU are also suffering from such poor behavior. Corrected results will be updated soon.

--ELU VS. tanh

![ELU VS. tanh](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/tanh-ELU.png)

--ELU VS. LeakyReLU

![ELU VS. LeakyReLU](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/LeakyReLU-ELU.png)

--Sigmoid VS. tanh

![sig VS. tanh](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/sig-tanh.png)

Sigmoid and tanh both show steady improving accuracy as epoch increases, with a fairly nice (around 90%) starting accuracy.

# Update Wed. July 11, 2018

## XNOR Gate by Simple 2-layer Neural Network:

>xnor_gate_sigmoid_TF.py

--Using sigmoid for the all layers' activation and square error as a cost function.

--Using AdamOptimizer for the neural network.

--XNOR = AND || (NOT x1)AND(NOT x2)

--Result prediction for [0, 0], [1, 0], [0, 1], [1, 1] and trained weights

![XNOR Result](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/XNOR-gate.png)

## Simple Generative Adversarial Networks:

>simple_GAN_tensorflow.py

--Approximating a 1-dimensional Gaussian distribution.

--Result (generated data-real data-decision boundary)

![GAN Result](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/simple_GAN_result.png)

