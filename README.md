# Various Neural Networks Projects by TensorFlow
This repository contains different kinds of Neural Networks Projects implemented in TensorFlow.

# Requirements
------------------- ---------
jupyter             1.0.0

Keras               2.2.0

Keras-Preprocessing 1.0.1 

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

# Update Thu. July 12, 2018
Exploring different performance of various activation functions on Neural Network:
NN_Activations_Comparison.py
Basic activation functions: sigmoid, relu, and tanh.
Advanced activation functions: LeakyReLU and ELU.

Test Case: Neural Network for solving the MNIST dataset recognition
http://yann.lecun.com/exdb/mnist/

Comparing Results (Model-Loss VS. Epoches): 

Sigmoid VS. ReLU VS. tanh

![Sigmoid VS. ReLU VS. tanh](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/sig-relu-tanh.png)

ELU VS. tanh

![ELU VS. tanh](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/tanh-ELU.png)

ELU VS. LeakyReLU

![ELU VS. LeakyReLU](https://github.com/cristianoBY/Neural-Networks-Projects-TensorFlow/blob/master/TF%20pics/LeakyReLU-ELU.png)

# Update Wed. July 11, 2018

1. XNOR Gate by simple 2-layer Neural Network:
xnor_gate_sigmoid_TF.py
Using sigmoid for the all layers' activation and square error as a cost function.
Using AdamOptimizer for the neural network.
XNOR = AND || (NOT x1)AND(NOT x2)

2. Simple Generative Adversarial Networks:
simple_GAN_tensorflow.py
Approximating a 1-dimensional Gaussian distribution.
