# Transfer Learning

**Transfer learning** is a technique that allows you to use features from existing, pre-trained models
- Instead of training from scratch, you can use a pre-trained model’s convolutional layers to extract features from your data.
- The **Inception model**, trained on **ImageNet** (1.4 million images, 1000 classes), is a strong example used in transfer learning.
- In the Inception model, you can lock the convolutional layers (which have already learned useful features) and retrain only the dense layers on your specific dataset.
- Alternatively, if some convolutional layers are too specialized for the original dataset, you can choose to retrain those as well.
- Trial and error is often needed to find the best combination of layers to lock or retrain.

## Coding transfer learning from the inception model

Bellow is the step-by-step process to code transfer learning using the inception model.

**Download Inception's Model Weights**

- Set the URL for the pre-trained weights of the InceptionV3 model (excluding the fully connected layers) and specify the local path where the weights will be saved. Use `tf.keras.utils.get_file` to download the weights from the URL.

```python
weights_url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
local_weights_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
tf.keras.utils.get_file(local_weights_file, weights_url)
```

**Create the Pre-Trained Model**

- The InceptionV3 model is instantiated using Keras, with `include_top=False` indicating that the fully connected layer at the top (used for classification) is excluded.
- The input shape is set to `(150, 150, 3)`, which means the model expects images with a size of 150x150 pixels and 3 color channels (RGB).
- `weights=None` ensures that no pre-trained weights are loaded at this stage; the model will be built but uninitialized until the downloaded weights are applied.

```python
pre_trained_model = tf.keras.applications.InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)
```

**Load the Downloaded Weights into the Model**

- The downloaded weights file is loaded into the pre-trained model.
- This step initializes the model's parameters using the pre-trained weights from the file.

```python
pre_trained_model.load_weights(local_weights_file)
```

**Freeze the Layers of the Pre-Trained Model**

- The code iterates through each layer of the pre-trained model and sets its `trainable` attribute to `False`, indicating that these layers should not be updated during training.
- Freezing layers is a common practice when performing transfer learning, allowing you to keep the learned features intact while focusing on training new layers that adapt to your specific task.

```python
for layer in pre_trained_model.layers:
    layer.trainable = False
```

**Print a Summary of the Model Architecture**

- The model summary is displayed, which provides an overview of each layer in the model, including layer types, output shapes, and the number of parameters.
- Since InceptionV3 is a large model, the summary will be extensive.

```python
pre_trained_model.summary()
```

## Coding Your Own Model With Transferred Features

**Defining the Last Layer**

- Inspecting the model summary reveals that the lower layers utilize convolutional operations with a kernel size of $3 \times 3$.
- To achieve richer feature extraction, the user selects the **mixed7** layer, which results from several convolutions with a $7 \times 7$ kernel size.
- The code is designed to extract the output from the mixed7 layer of the Inception model, rather than the last layer.

```python
# Get the output of the mixed7 layer
last_layer = pre_trained_model.get_layer('mixed7')
last_ouput = last_layer.output
```

**Defining a New Model**

- The new model is defined using the output from the **mixed7** layer, referred to as `last_output`.
- This model structure is similar to dense models created earlier in the course, with some differences in implementation.
- The output from the mixed7 layer is flattened to prepare it for the dense layers.
- A dense hidden layer with **1024 neurons** and a **ReLU** activation function is added.
- An output layer with a single neuron uses a **Sigmoid** activation function for binary classification.
- The new model is created using the `Model` abstract class, passing the input from the pre-trained model and the defined layers.
- The model is compiled with a specified optimizer, loss function, and performance metrics.

```python
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

# Flatten the output from the mixed7 layer
x = tf.keras.layers.Flatten()(last_output)

# Add a dense hidden layer
x = tf.keras.layers.Dense(1024, activation='relu')(x)

# Add an output layer with a Sigmoid activation function
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create the new model
model = tf.keras.Model(inputs=pre_trained_model.input, outputs=output_layer)

# Compile the model
model.compile(
    optimizer=RMSprop(learning_rate=0.0001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)
```

## Exploring Dropouts

- We observed that the model performs exceptionally well on the **training data**.
- However, it struggles to generalize to new, unseen data, as indicated by the **validation set**.
- The validation accuracy begins at approximately **94%** but displays a noticeable downward trend over time.
- This increasing gap between training and validation accuracy is a classic sign of **overfitting**.
- Despite utilizing transfer learning and data augmentation, we still encountered this issue.
- To address overfitting, we introduced a **dropout layer** in Keras:
  - Dropout helps mitigate overfitting by **randomly dropping units** during training.
  - This technique reduces the risk of layers with **similar weights** influencing each other.
- Here is the updated code to include the dropout layer:

```python
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

# Flatten the output from the mixed7 layer
x = tf.keras.layers.Flatten()(last_output)

# Add a dense hidden layer
x = tf.keras.layers.Dense(1024, activation='relu')(x)

# Add a dropout layer
x = tf.keras.layers.Dropout(0.2)(x)

# Add an output layer with a Sigmoid activation function
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create the new model
model = tf.keras.Model(inputs=pre_trained_model.input, outputs=output_layer)

# Compile the model
model.compile(
    optimizer=RMSprop(learning_rate=0.0001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)
```

- After analyzing the accuracy charts with the dropout layer incorporated, we found a **significant improvement** in accuracy.