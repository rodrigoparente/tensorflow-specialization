# Introduction to Computer Vision

- Computer vision is the field where computers are trained to recognize and label objects in images.
- While humans can easily identify objects like shirts or shoes, it's challenging to manually program a computer to do the same.
- The key challenge is teaching a computer to recognize and differentiate objects using visual data.
- A common solution is to provide the computer with many labeled images, allowing it to learn the patterns that distinguish one object from another.

**Fashion MNIST Dataset**

- The **Fashion MNIST dataset** includes 70,000 images of clothing items, grouped into 10 categories.
- Each image is resized to **28x28 pixels**, small enough to reduce the computer's processing load but large enough to preserve important features for recognizing objects.
- The images are in **grayscale**, with pixel values ranging from **0 to 255**, meaning each pixel is represented by one byte.
- Even with reduced size and grayscale, the images still provide enough detail for both humans and computers to recognize objects, like an ankle boot.

**Loading the Dataset**:

  - Fashion MNIST is conveniently available via TensorFlow’s API, so there’s no need to manually handle image files.
  - Use the `fashion_mnist` dataset from TensorFlow’s Keras database, which provides a simple way to load the data.

  ```python
  import tensorflow as tf
  
  # Load the Fashion MNIST dataset
  fashion_mnist = tf.keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  ```

## Coding a Computer Vision Neural Network

- We can define a neural network to classify clothing items in the Fashion MNIST dataset with the following layers:
  1. **Input Layer**: Accepts images of size $28 \times 28$.
  2. **Flatten Layer**: Converts the $28 \times 28$ image into a linear array of 784 values.
  3. **Hidden Layer**: Contains **128 neurons** that compute weighted sums of the inputs, represented as:
     $$ y = w_1 x_1 + w_2 x_2 + \ldots + w_{128} x_{128} $$
     - Activation function: **ReLU** (`tf.nn.relu`).
  4. **Output Layer**: Contains **10 neurons**, each corresponding to one of the 10 clothing classes.
     - Activation function: **softmax** (`tf.nn.softmax`).

- The code to implement this neural network is as follows:

```python
import tensorflow as tf
from tensorflow.keras import models

# Define the neural network model
model = models.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

## Using Callbacks to control training

- The training loop supports **callbacks**, enabling users to run a function at the end of each epoch to check performance metrics and potentially stop training.  
- Callbacks can be implemented as a separate class, but they can also be included inline with other code.  
- The main function to focus on is `on_epoch_end`, which is called at the end of each epoch.  
- This function receives a `logs` object that contains important training metrics.  
- Below is an example of how to implement a callback.  
  
```python
import tensorflow as tf

# Define the callback class
class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('loss') < 0.4:
      print(f"Loss is low so cancelling the training!")
      self.model.stop_training = True

...

# Start training with the callback
model.fit(train_images, train_labels, epochs=100, callbacks=[MyCallback()])
```