# Various small projects implemented in TensorFlow
This repository contains various small projects implemented in TensorFlow.

# Update Thu. Aug 16, 2018

## LSTM neural network for three-class classification problem

>LSTM_NN

The LSTM neural network is used to deal with an emotion classification problem. The dataset used is part of the SJTU Emotion EEG Dataset (SEED), i.e. a three-class classification problem and the features extracted from emotional EEG signals. 

The dataset can be downloaded from: http://bcmi.sjtu.edu.cn/~seed/index.html. Please refer to link http://bcmi.sjtu.edu.cn/~seed/description.html to read more about the experimental procedure. During the experiment, the participants were required to watch fifteen movie clips of different emotion categories.

Four files are provided: three .npz files and one .npy file. Both .npz files and .npy files can be loaded with numpy.load. Three .npz files are data collected from three subjects. There are 15 numpy ndarray files in every .npz file, corresponding to 15 movie clips. Each numpy ndarray is of shape (62 * n * 5), where 62 is channel number, n is sample number, and 5 is frequency bands.

I first built LSTM classification models individually, i.e. one model for each of the three .npz file provided. During watching movie clips, emotions of subjects would change with time. LSTMs are suitable to capture this temporal information.

Since EEG signals are different for different people, and this might cause trouble in building an universal emotion model for different people. So I built another LSTM model with all .npz files provided, and compare the classification results with previous results.

## Convolutional neural network for MNIST

>CNN_MNIST

Convolutional neural network (CNN) for multi-class classification problems.

The dataset used in this homework is the MNIST database. Including:

train-images-idx3-ubyte.gz: training set images (9912422 bytes)
train-labels-idx1-ubyte.gz: training set labels (28881 bytes)
t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)
t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)

Solved the ten-class classification problem in the given dataset using feed-forward neural network. Also visualize the deep features extracted before feed-forward layers.

Reference: Lecun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998, 86(11):2278-2324.

## SVM and the Min-Max-Module for multi-class classification problem

>SVM_SEED

The dataset used is part of the SJTU Emotion EEG Dataset (SEED) (Please refer to LSTM_NN above). Four files are provided: train_data.npy, train_label.npy, test_data.npy, and test_label.npy. 37367 samples are included in the training data, and 13588 samples in the test data.

Solved the three-class classification problem in the given dataset using SVM classifiers. Wrote my own one-vs-rest strategy.

Solved the three-class classification problem using Min-Max-Module SVM and part-vs-part task decomposition method. Divided the three-class problem into three two-class problems using one-vs-rest method and then decompose these imbalanced two-class problems.

# Update Thu. July 26, 2018

## Convolutional Neural Network for Google QuickDraw Game

>QuickDraw

Convolutional Neural Network for a QuickDraw classifier trained by QuickDraw dataset provided by Google

https://console.cloud.google.com/storage/browser/quickdraw_dataset/?pli=1

I used the simplified data in form of numpy bitmap provided by Google and the Quick Draw AI Experiments

https://quickdraw.withgoogle.com/data

Check

>Quick_Draw_New.ipynb

for Final Version trained in Google Colab 

(3 examples were tests instead of 100 for time-saving, please remember to change the mini-class.txt and size of softmax layer!)

![QD](https://github.com/cristianoBY/TensorFlow-Project/blob/master/QuickDraw/Screen%20Shot%202018-08-12%20at%205.35.36%20PM.png)

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

