# ResNet-50 Transfer Learning the Cats and Dogs Dataset

Using [TensorFlow] API to transfer a [ResNet-50] neural network and connect new fully connected layers to classify [cats and dogs images].

---

## Problem Overview

Classifying cats and dogs images has been a popular task in the deep learning field. There are many different "Cats and Dogs" datasets, but the objective is the same: train a convolutional neural network (CNN) able to successfully differentiate cats from dogs.

![cat 14](https://user-images.githubusercontent.com/33037020/192171611-5a9c3ba6-190a-4bb3-b10c-80fcbc123b89.jpg) ![dog 35](https://user-images.githubusercontent.com/33037020/192171834-eb837005-1c43-4157-b09d-28e3b06da35b.jpg)

Our version of the dataset has [cats and dogs images] already separated in training and testing folders. Thus, we can use TensorFlow to create a data generator with a validation set split and focus on training the dense layers of the model.

[Transfer learning] consists of copying the weights and architecture of a network, while maintaining or retraining some of its layers for particular needs. It is recurrently used to save model building time by using weights from models already trained in other more general datasets. In our case, cats and dogs are our classes, which are also part of the more general [ImageNet] dataset. This means that we can pick any CNN trained using ImageNet to get a warm start at training our own model.

## Analysis Introduction

[ResNet-50] is a somewhat old, but still very popular, CNN. Its popularity come from the fact that it was the CNN that introduced the residual concept in deep learning. It also [won the ILSVRC 2015] image classification contest. Since it is a well-known and very solid CNN, we decided to use it for our transfer learning task.

As the original ResNet-50 was trained on ImageNet, its last layer outputs 1000 probabilities for a tested image to belong to the 1000 different ImageNet classes. Therefore, we cannot directly use it in our binary classification problem with only cats and dogs as classes. Nonetheless, using the original weights of the network would give us a model that is too generalistic and not really built to understand cats and dogs.

We first transfer a base ResNet-50 CNN, that is, a ResNet-50 without its fully connected layers. Later, by freezing the base ResNet-50 weights, we add new layers and train them without changing anything in the convolutional section of the network. In this case, the convolutional section becomes just an image feature extractor and the actual job of classifying the features is performed by the newly added fully connected layers.

After many experiments, an optimal architecture was found. **It achieves 99% accuracy, precision and recall in our cats and dogs testing set.** In this project's notebook, we show how to build and train this CNN. Next there is a grid with 25 random cats and dogs images visually showing how our model predicts the testing data:

![cats_and_dogs_grid](https://user-images.githubusercontent.com/33037020/192171660-f116523e-f4d3-40f3-8301-acde1a434720.PNG)

[//]: #

[cats and dogs images]: <https://www.kaggle.com/datasets/tongpython/cat-and-dog>
[ImageNet]: <https://www.image-net.org>
[won the ILSVRC 2015]: <https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8>
[TensorFlow]: <https://www.tensorflow.org>
[ResNet-50]: <https://en.wikipedia.org/wiki/Residual_neural_network>
[Transfer learning]: <https://en.wikipedia.org/wiki/Transfer_learning>
