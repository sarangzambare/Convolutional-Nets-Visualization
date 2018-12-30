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


To visualize how convolutional networks "see" things, I used the original pre-trained vgg16 network,
and produced heatmaps of the class which an example photo maximally activated to. Below is the example image and the heatmap:

### Example Image:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/example_dog.png)


This image was classified as "Labrador Retriever" with 69.9% accuracy. Below is the heatmap of the example image over the class "Labrador Retriever" 

### Heatmap over Labrador Retriever class:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/heatmap.png)



This gives a slight insight into what led the network into calling the image as a "labrador retriever". (Its the ears!)


I also plotted the shallower and deeper layers of the network, on an example image (a cat), just for the funzies.

### Example Image:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/example_cat.png)


The activations for the image above for various filters of various layers of the network are given in the below grid. As expected, we see that the deeper the layer is, the more "specific" features it tries to identify (like whiskers, or sharp vs not sharp years etc):

### Activations for different filters| shallow layers:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/grid1.png)


### Activations for different filters| deeper layers:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/grid2.png)





