# Hands-On: Convolutional Neural Networks with PyTorch


Convolutional neural networks revolutionized computer vision and allowed the increase of performance in multiple tasks of computer vision such as classification, segmentation, object detection, etc. They were a breakthrough in a wide range of applications and have been present in state of the art algorithms for over a decade. In this hands-on you will go through the experimentation process of training convolutional neural networks performing changes on the architecture and the training parameters to enhance the performance of the model in image classification.


## Table of contents
- [1. Learning Objectives](#1-learning-objectives)
- [2. Tech Stack](#2-tech-stack)
- [3. Instructions](#4-instructions)
 - [3.1. Dataset](#31-dataset)
 - [3.2. Model](#32-model)
 - [3.3. Training Pipeline](#33-training-pipeline)
 - [3.4. Test file](#33-test-file)
 - [3.3. Demo file](#33-demo-file)
- [4. Deliverables](#4-deliverables)


## 1. Learning Objectives


After completing this workshop, trainees will be able to:


- Describe the challenges to design the proper convolutional architecture.
- Write a pipeline for training convolutional models for classification using PyTorch.
- Train a neural network aiming to tackle the problem of natural language translation.


## 2. Tech Stack
- PyTorch
- Caltech101
- GCP


## 3. Instructions


In this workshop you will be exploring the use of convolutional neural networks for image classification. This will be performed in three main steps for which you will have to develop code following the TODOs in each module pointed by this readme. The main goal here is for you to design your own ConvNet architecture that reaches the best performance possible in image classification


### 3.1. Dataset


You will train your neural network using Caltech101, a classic dataset collected in September 2003 by Fei-Fei Li, Marco Andreetto, and Marc'Aurelio Ranzato that contains pictures of objects that belong to 101 categories, with about 40 to 800 images per category. The size of each image is roughly 300x200 pixels, so take this into account when designing your architecture.


The module datasets.Dataloader contains the code to properly load Caltech101 as a custom dataset using PyTorch. This module contains the Caltech101 class required to load as an iterable. Complete the TODOs in that file implementing the:


- class attributes
- _load_image function
- _load_label function
- __getitem__ function


### 3.2. Model


In this phase you will implement your custom convolutional model from scratch. The models.cnn module contains the implementation of AlexNet from PyTorch that you can use for inspiration. You can build over this network and modify the architecture (adding layers, skip connections, non-linearities, etc) to improve performance. This is an iterative process to discover the proper architecture so don't be discouraged by negative results. Take wild risks when implementing your own custom network to achieve good performance. This [paper](https://arxiv.org/abs/1407.1610) might give you additional insights on what to consider when designing your own network.


### 3.3. Training Pipeline


The train.py module contains the structure required to implement a proper training pipeline in PyTorch for this task. Complete the TODOs in this module, modify the hyperparameters and select the proper loss and optimizer setups to be able to train your network and achieve good performance. Take into account that this module uses hydra to provide a hierarchical configuration by composition to the module so use the config files provided in this repo to achieve your goal.

### 3.4 Test file

Create a test file that loads the weights of the model and save/loads the appropiate metrics of you trained model

### 3.5 Demo file

Inclure in your workshop a demo file that will allow to perform inference, returns the class of a test image and allows the further deployment of the model.

## 4. Deliverables


- datasets.Dataloader module complete
- models.cnn module complete with your own custom ConvNet implementation
- train.py module complete
- test.py for metrics assessment
- demo of you neural network for further usage


## 5. References


- [1] F.-F. Li, M. Andreeto, M. Ranzatoand P. Perona, “Caltech 101”. CaltechDATA, Apr. 06, 2022. doi: 10.22002/D1.20086.
- [2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, ‘ImageNet Classification with Deep Convolutional Neural Networks’, in Advances in Neural Information Processing Systems, 2012, vol. 25.