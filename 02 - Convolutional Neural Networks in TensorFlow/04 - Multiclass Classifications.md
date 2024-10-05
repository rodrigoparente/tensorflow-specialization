# Moving From Binary to Multi-class Classification

- We will use the Rock-Paper-Scissors dataset that containing 2,892 images of hands in Rock/Paper/Scissors poses.
- Images were generated using CGI with diverse models, including male and female with various skin tones.
- The dataset contains training set, validation set, and extra images for testing.

## Loading Dataset

- For binary classification:

```python
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)
```

- For multiple classes:

```python
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(150, 150),
    batch_size=32,
    label_mode='categorical'
)
```

## Model Definition

- For binary classification, we use one output neuron with a sigmoid activation function:

```python
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
```

- For multiclass classification, we need to change output layer to have three neurons (for rock, paper, and scissors).
- Also we change the activation from `sigmoid` to `softmax`, so get the probability for each class:

```python
output_layer = tf.keras.layers.Dense(3, activation='softmax')(x)
```

## Network Compilation

- When compiling a model for binary classification, we use the `binary_crossentropy` loss function:

```python
    model.compile( 
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
```

- For multi-class classification we can either use the `categorical_crossentropy`, when you have one-hot encoded labels:

```python
model.compile( 
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
```

- Or `sparse_categorical_crossentropy`when your labels are already in integer format, which can be more memory-efficient:

```python
model.compile( 
    optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
```
