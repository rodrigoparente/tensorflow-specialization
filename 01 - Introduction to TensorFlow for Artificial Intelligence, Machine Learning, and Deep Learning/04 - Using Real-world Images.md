# Understanding the tf.data API

- In real-world scenarios, data is often unsplit and lacks labels.
- The `tf.data` API simplifies loading, preprocessing, and managing large datasets by creating efficient pipelines for seamless integration from various sources. 
- Features like caching, shuffling, and prefetching optimize memory and computation, accelerating deep learning model training.

## Loading and Preprocessing Data with TensorFlow

- You can easily load images and generate labels from subdirectory names using the Keras function `image_dataset_from_directory`.
- For example, consider this directory structure:
    - `/images/train/horses/` (horse images)
    - `/images/train/humans/` (human images)
- The following code loads and preprocesses the images:

```python
import tensorflow as tf

# Load images from the training directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(300, 300),
    batch_size=128,
    label_mode='binary'
)
```

- Key points:
  - Images are automatically resized to a uniform size (e.g., 300x300) since neural networks require consistent input dimensions.
  - Images are loaded in batches (e.g., 128 images per batch), improving efficiency over single-image loading.
  - In this case, binary labels (0 for horses, 1 for humans) are used, specified with `label_mode='binary'`.

## Data Pipeline Optimization

- Consider this Python code for optimizing the data pipeline:

```python
import tensorflow as tf

# Apply normalization
rescale_layer = tf.keras.layers.Rescaling(scale=1./255)
train_dataset_scaled = train_dataset.map(
    lambda image, label: (rescale_layer(image), label))

train_dataset_final = (train_dataset_scaled
                       .cache()
                       .shuffle(buffer_size=1000)
                       .prefetch(buffer_size=tf.data.AUTOTUNE))
```

**Normalization**

- The `Rescaling` layer scales pixel values from 0 to 1 for easier processing by the neural network.

**Data Caching**

- `cache()` stores the dataset in memory after the first pass, speeding up subsequent access during training.

**Shuffling**

- `shuffle()` randomizes the order of data using a buffer (e.g., 1000 images), preventing the model from learning any unintended sequence in the data.

**Prefetching**

- `prefetch()` allows the model to load the next batch while processing the current one. Using `tf.data.AUTOTUNE` optimizes the buffer size automatically for better performance.

## New Network Architecture
Here’s an analysis of the neural network architecture based on your code:

## Implementing Convolutional Layers

- Consider the following code that defines a convolutional neural network:

```python
import tensorflow as tf

model = tf.keras.models.Sequential([ 
    tf.keras.Input(shape=(300, 300, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
- And the model summary below:

| Layer (type)                       | Output Shape         | Param #   |
|------------------------------------|----------------------|-----------|
| conv2d_5 (Conv2D)                  | (None, 298, 298, 16) | 448       |
| max_pooling2d_5 (MaxPooling2D)     | (None, 149, 149, 16) | 0         |
| conv2d_6 (Conv2D)                  | (None, 147, 147, 32) | 4640      |
| max_pooling2d_6 (MaxPooling2D)     | (None, 73, 73, 32)   | 0         |
| conv2d_7 (Conv2D)                  | (None, 71, 71, 64)   | 18496     |
| max_pooling2d_7 (MaxPooling2D)     | (None, 35, 35, 64)   | 0         |
| flatten_1 (Flatten)                | (None, 78400)        | 0         |
| dense_2 (Dense)                    | (None, 512)          | 40,141,312|
| dense_3 (Dense)                    | (None, 1)            | 513       |


**Input Layer**

- The input shape `(300, 300, 3)` corresponds to color images of size **300x300** with 3 channels (RGB).

**First Conv2D Layer (conv2d_5 + max_pooling2d_5)**

- **Conv2D layer** applies **16 filters** of size **3x3**, reducing the dimensions from **300x300** to **298x298** because the filter skips the border pixels.
- The **MaxPooling2D** layer halves the dimensions, reducing the size to **149x149**.

**Second Conv2D Layer (conv2d_6 + max_pooling2d_6)**

- The next **Conv2D layer** applies **32 filters** of size **3x3**, reducing the size from **149x149** to **147x147**.
- The **MaxPooling2D** layer further reduces the size to **73x73**.

**Third Conv2D Layer (conv2d_7 + max_pooling2d_7)**

- The final **Conv2D layer** applies **64 filters** of size **3x3**, shrinking the size from **73x73** to **71x71**.
- The **MaxPooling2D** layer cuts the size down to **35x35**.

**Transition to Dense Layers (flatten_1)**

- The output from the last convolutional layer is **64 feature maps** of size **35x35**.
- After flattening, the output shape becomes:
- The flattened data is fed into a **Dense** layer with **512 neurons** and a **ReLU activation** function.
- The final **Dense layer** uses **1 neuron** with a **Sigmoid activation** function, ideal for **binary classification**.

## Compiling the ConvNet

- Consider the following code:

```python
import tensorflow as tf

model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), 
              metrics=['accuracy'])
```

- For binary classification, we use **binary cross-entropy** as the loss function, which measures the performance of a model whose output is a probability value between 0 and 1.
- We choose the **RMSprop** optimizer that allows for adjusting the learning rate, which can help in experimenting with model performance.

## Training the ConvNet

- Consider the following code:

```python
history = model.fit(training_dataset_final, 
                    epochs=15, 
                    validation_data=validation_dataset_final, 
                    verbose=2)
```

- We utilize a **TF.data.dataset** instead of traditional NumPy arrays for training data.
- The training dataset contains both the images and their respective labels, so there's no need to specify labels separately.
- We set the number of epochs to **15**, which indicates the number of complete passes through the training dataset.
- We also include a validation set created earlier for evaluating the model's performance during training.
- The verbose was set to **2**, this parameter controls the amount of information displayed during training.

## Model Prediction

- After training the model, we want to perform predictions on new data.
- For this, consider the following code:

```python
from google.colab import files

uploaded = files.upload()

for filename in uploaded.keys():
    # predicting images
    path = '/content/' + filename
    image = tf.keras.utils.load_img(path, target_size=(300, 300))
    image = tf.keras.utils.img_to_array(image)
    image = rescale_layer(image)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image, verbose=0)[0][0]
    print(f'\nmodel output: {prediction}')
    
    if prediction > 0.5:
        print(filename + " is a human")
    else:
        print(filename + " is a horse")
```

- The `model.predict()` function is called with the prepared image(s).
- For binary classification, the model returns a single value in the predictions array:
    - Close to **0** for class 0 (negative).
    - Close to **1** for class 1 (positive).

## Rerences

 - [Binary Crossentropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
 - [Optimizers](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
 - [RMSprop](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)
 - [Binary CLassification](https://youtu.be/eqEc66RFY0I)