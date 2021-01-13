# Hand-Written-Text-Recognition
This project includes an implementation of pytorch library, named Mytorch. The library is used to construct a fully connected neural network for hand-written text recognition. 
## Introduction
Mytorch is a partial implementation of Pytorch library. It includes a framework of automatic differentiation (autograd). During forward propagation, autograd will automatically construct a computational graph. During back propagation, the gradients for operations that require gradient will be calculated automatically through recursion. Other functionalities of Mytorch includes: ReLU activation, cross entropy loss function, and SGD optimizer. A neural network was constructed using Mytorch and was trained on MNIST dataset.

Another portion of this project is an image processing function. In this portion a hand-written note is given and the function will identify and crop texts from it. The cropped texts are arranged by the same order as they appear in the image. 
