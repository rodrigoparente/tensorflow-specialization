# Real-world Time Series Data

Bellow is the code for the neural network that includes the convolutional layer, as well as LSTMs.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 200)
])
```

The learning rate changes each epoch through a learning rate scheduler.

```python
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
```

The optimizer is set to stochastic gradient descent (SGD) with a momentum of 0.9, and an initial learning rate of $1e^{-8}$.
 
```python
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
```

The model is compiled with the optimizer defined before and with the Huber loss.

```python
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
```

Training the model for 500 epochs revealed a learning rate sweet spot around $10^{-5}$.
  
```python
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
```

The model performed well but could benefit from further fine-tuning to address remaining instability and overfitting issues. For example, the batch size can be adjusted for further experimentation

