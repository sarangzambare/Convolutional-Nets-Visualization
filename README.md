# Visualizing what convnets learn with a simple dataset
## Author: Sarang Zambare


This is a project that I undertook to visualize what exactly does a convolutional
neural network "see" when it tries to classify images.

The dataset that I used was a very basic one: https://www.kaggle.com/c/dogs-vs-cats/data
It contains a huge number of labelled images of cats and dogs from various perspectives and in different
backgrounds.

I used Keras for most of the learning and parameter tuning for this project. Using a basic alternation of
8 maxpool and convolutional layers, I was able to achieve an 78% validation accuracy.

I tried using feature extraction using the vgg16 network, with a final dropout of 50% (before flattening), and I was able to up the validation accuracy to about 85%

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/png/trainval_loss.png)


To visualize how convolutional networks "see" things, I used the original pre-trained vgg16 network,
and produced heatmaps of the class which an example photo maximally activated to. Below is the example image and the heatmap:

### Example Image:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/png/example_dog.png)


This image was classified as "Labrador Retriever" with 69.9% accuracy. Below is the heatmap of the example image over the class "Labrador Retriever"

### Heatmap over Labrador Retriever class:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/png/heatmap.png)



This gives a slight insight into what led the network into calling the image as a "labrador retriever". (Its the ears!)





I also plotted the shallower and deeper layers of the network, on an example image (a cat). For reference, the summary of the entire network is shown below:

```

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6272)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0

```

### Example Image:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/png/example_cat.png)


The activations for the image above for various filters of various layers of the network are given in the below grid. As expected, we see that the deeper the layer is, the more "specific" features it tries to identify (like whiskers, or sharp vs not sharp years etc):

### Activations for different filters| shallow layers:

```
images_per_row = 16

for layer_name, layer_activation in zip(layer_names,activations):
    n_features = layer_activation.shape[-1]    # Number of features in this layer
    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size*n_cols, images_per_row*size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:,:,col*images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='plasma')

```    

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/png/grid1.png)


### Activations for different filters| deeper layers:

![alt text](https://raw.githubusercontent.com/sarangzambare/cats_vs_dogs/master/png/grid2.png)
