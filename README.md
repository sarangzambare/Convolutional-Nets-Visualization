# Visualizing what convnets learn with simple dataset
## Author: Sarang Zambare


This is a project that I undertook out to visualize what exactly does a convolutional
neural network "sees" when it tries to classify images.

The dataset that I used was a very basic one: https://www.kaggle.com/c/dogs-vs-cats/data
It contains a huge number of labelled images of cats and dogs from various perspectives and in different
backgrounds.

I used Keras for most of the learning and parameter tuning for this project. Using a basic alternation of 
8 maxpool and convolutional layers, I was able to achieve an 78% validation accuracy.

I tried using feature extraction using the vgg16 network, with a final dropout of 50% (before flattening), and I was able to up the validation accuracy to about 85%

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/trainval_loss.png)





