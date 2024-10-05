# What are convolutions and pooling?

**Convolution**

- Convolution is a process that applies a filter (or kernel) to an image to extract important features.
- This operation helps emphasize specific patterns like **edges**, **lines**, or **textures**.
- Different filter sizes (e.g., **3x3**, **5x5**) can be used to detect various features.

**Pooling**

- A downsampling technique that compresses the image, significantly reducing its dimensions.
- This process lowers computational complexity while preserving important features.

## Implementing Convolutional Layers

- Here's the previous neural network code, modified to include convolutional and pooling layers:
  
```python
import tensorflow as tf

model = models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

- The input shape `(28, 28, 1)` corresponds to grayscale images that are **28x28** pixels in size, where `1` indicates a single color channel (grayscale).
- The **Conv2D** layer generates **64 filters** of size **3x3** and uses the **ReLU** activation function, which eliminates negative values from the output.
- The **MaxPooling2D** layer reduces the image dimensions through **2x2 max pooling**, which selects the highest value from each **2x2** pixel grid.

## Implementing Pooling Layers

- The model summary method is useful for inspecting the layers of the model.
- It provides a table displaying the layers and their details, including the output shape.

| Layer (type)                       | Output Shape         | Param #  |
|------------------------------------|----------------------|----------|
| conv2d_12 (Conv2D)                 | (None, 26, 26, 64)   | 640      |
| max_pooling2d_12 (MaxPooling)      | (None, 13, 13, 64)   | 0        |
| conv2d_13 (Conv2D)                 | (None, 11, 11, 64)   | 36,928   |
| max_pooling2d_13 (MaxPooling)      | (None, 5, 5, 64)     | 0        |
| flatten_5 (Flatten)                | (None, 1600)         | 0        |
| dense_10 (Dense)                   | (None, 128)          | 204,928  |
| dense_11 (Dense)                   | (None, 10)           | 1,290    |

**First Conv2D Layer (conv2D_12)**

- The output shape may appear confusing, particularly when transitioning from an input size of 28x28 to 26x26.
- This discrepancy is due to the use of a 3x3 filter, which cannot be calculated on border pixels lacking sufficient neighbors.
- Calculations can only begin on pixels that have all eight neighbors (e.g., the first pixel that meets this criterion).
- As a result, the output dimensions shrink by 2 pixels in both the X and Y directions when using a 3x3 filter.

**First MaxPooling Layer (max_pooling2d_12)**

- The first max pooling layer reduces the size from 26x26 to 13x13.

**Second Conv2D Layer (conv2D_13)**

- The second convolution layer reduces the output dimension by 2, leaving the image with 11x11 pixels.

**Second MaxPooling Layer (max_pooling2d_1)**

- The second max pooling layer further reduces the size from 11x11 to 5x5 images.

**Input to Dense Neural Network (flatten_5)**

  - The dense neural network now receives 5x5 images instead of the original 28x28 images.
  - The network processes multiple convolutions, in this case, 64 convolutions, producing 64 images of size 5x5.
  - After flattening, the output consists of 25 pixels per image (5x5) times 64 filters, totaling 1600 elements.
  - This is significantly larger than the original 784 pixels from the 28x28 input.


# References

- [Convolutional Neural Networks](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)