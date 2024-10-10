# Recurrent Neural Networks for Time Series

- A Recurrent Neural Network (RNN) is designed to process sequences of inputs through recurrent layers.
- RNNs are flexible and capable of handling various types of sequences, such as text and time series data.

**Memory Cells**

- RNN layers consist of memory cells that are reused at each time step to compute outputs.
- At each time step:
    - The memory cell receives the input value for that step (e.g., zero at time zero) and the previous state (zero state input).
    - It calculates the output (e.g., $Y_0$) and a state vector ($H_0$) to be fed into the next time step.
    - $H_0$ is combined with the input at time step 1 ($X_1$) to produce output $Y_1$ and state $H_1$, continuing until all input dimensions are processed.
- This feedback loop is what makes the architecture a recurrent neural network, as the output from one time step is fed back into itself in the next time step.

**Significance**

- The recurrence of values helps in understanding dependencies in the data.
- For example, the position of a word in a sentence can influence its meaning; similarly, the proximity of numbers in a series may affect their influence on the target value.

## Shape of the Inputs to the RNN

- The input data for RNNs is three-dimensional.
- For example, with a window size of 30 time steps, a batch size of 4, and considering a univariate time series, the input shape is $4 \times 30 \times 1$.

**Memory Cell Input**

- In this example, at each time step, the memory cell receives
    - A $4 \times 1$ matrix as the current input.
    - The state matrix from the previous time step, which is zero for the first step and the output from the memory cell for subsequent steps.
- The memory cell outputs a Y-value matrix.
- If the memory cell contains $N$ neurons, the output matrix shape is $B \times N$ (where $B$ is the batch size and $N$ is the number of neurons).

**State Output**

- In a simple RNN, the state output $H$ is a direct copy of the output matrix $Y$:
    - $H_0$ is a copy of $Y_0$
    - $H_1$ is a copy of $Y_1$
    - This pattern continues for subsequent time steps.
- Each memory cell processes both the current input and the previous output at each time step.

**Sequence-to-Vector RNN**

- Sometimes, you may want to input a sequence but only receive a single output vector for each instance in the batch.
- This is referred to as a sequence-to-vector RNN, where all outputs except the last one are ignored.
- In Keras with TensorFlow, this behavior is the default.
- To have a recurrent layer output a sequence rather than just the last output, you must specify `return_sequences=True` when creating the layer.
- This is necessary when stacking multiple RNN layers.

## Outputting a Sequence

- In many applications, especially in time series forecasting, natural language processing, and sequence prediction tasks, we need to handle sequential data where each input may have a temporal or sequential relationship.
- Outputting sequences allows the model to learn and predict based on this temporal context.
- Bellow is an example of an RNN outputing a sequence

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size, 1)),
    tf.keras.layers.SimpleRNN(20, return_sequences=True),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])
```

**Return Sequences Parameter**

- The `return_sequences` parameter in RNN layers dictates whether the layer will output a sequence or just the last output. 
**`return_sequences=True`**

    - The layer returns the output for every time step in the input sequence.
    - This is useful when the subsequent layer also expects sequences as input, enabling deeper recurrent architectures (e.g., stacked RNNs).
    - For example, if you have an input sequence of length 10, the output will also be a sequence of length 10.
**`return_sequences=False`** (default behavior)

    - The layer returns only the output for the last time step of the input sequence.
    - This is useful when you need a summary of the entire input sequence, often for classification or regression tasks.
    - For example, given the same input sequence of length 10, the output will be a single value.

**Dimensionality of Outputs**

- The output dimensionality of an RNN layer depends on:
    **Input Shape**: Defined as `(timesteps, features)`

    **Number of Units**: Each RNN layer has a specified number of units (neurons). This defines how many values will be
    output for each time step when `return_sequences=True`.
    - For example, if an RNN layer has 20 units and `return_sequences=True`, the output shape will be `(batch_size, timesteps, 20)`.
    **Final Layer**: If a Dense layer follows the RNN layer, it will convert the last output (when
    `return_sequences=False`) to the desired output shape, which could be a single value for regression tasks or multiple classes for classification.

## Lambda Layers

- A Lambda layer allows you to add custom operations within the model definition.
- It helps to expand the model’s functionality by performing arbitrary computations.
- For example, we can scaling the outputs from the previous example to improve the training process.
- We just need to multiply the output values by 100 to align them with the expected value range of the time series data.
- The code below shows how to do that:

```python
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size, 1)),
    tf.keras.layers.SimpleRNN(20, return_sequences=True), 
    tf.keras.layers.SimpleRNN(20, return_sequences=True),  
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100)
])
```

## Adjusting the learning rate dynamically

- One effective approach is to adjust the hyperparameters of the model during training.
- In the following steps, we will train a Recurrent Neural Network (RNN) specifically for time series prediction, focusing on optimizing the model's performance by using a tailored learning rate and an appropriate loss function.

**Input Data Preparation**

  - To match the expected input format of the RNN layer, the dataset needs to have its dimensions expanded. This is done in the `windowed_dataset` function.
  
  ```python

  train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
  ```

**RNN Architecture**

- The model consists of two RNN layers, each with 40 cells.
- The input shape is designed to accommodate a window size of 1, including an additional dimension for compatibility with RNN layers.

```python
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size, 1)), 
    tf.keras.layers.SimpleRNN(40, return_sequences=True), 
    tf.keras.layers.SimpleRNN(40), 
    tf.keras.layers.Dense(1), 
    tf.keras.layers.Lambda(lambda x: x * 100.0) 
])
```

**Learning Rate Optimization**

- A learning rate scheduler is set up to adjust the learning rate dynamically during training. This helps to find an optimal learning rate quickly and saves time on hyperparameter tuning.

```python
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20)
)
```

**Optimizer Setup**

- The model is compiled using the Stochastic Gradient Descent (SGD) optimizer with an initial learning rate, and the Huber loss function is specified.
- The [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) function is employed, because it is less sensitive to outliers and is beneficial for noisy data commonly encountered in time series.

```python

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),
            optimizer=optimizer,
            metrics=["mae"])
```

**Training Configuration**

- The model is trained for 100 epochs, allowing for monitoring of loss and mean absolute error (MAE) metrics.

```python
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```

## Coding LSTMs

- LSTMs can remember information over longer periods, unlike RNNs, which struggle with long-term dependencies.
- They have a more complex structure with gates (input, output, forget) to manage information flow, whereas RNNs have a simpler design.  
- LSTMs are more stable during training and better suited for longer sequences compared to RNNs. 
- They excel in tasks requiring long-term context, such as language modeling and time series prediction, while RNNs are better for shorter sequences. 
- Below is a Python code to use LSTMs to create a DNN.

**Building the LSTM Model**

- Create a model with a two LSTM layer that contains 32 cells, making it bidirectional to assess its impact on predictions.
- Also, we set `return_sequences=True` in the first LSTM layer to allow the second LSTM layer to receive sequence data.

```python
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size, 1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), return_sequences=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])
```

**Model Compilation**

- Set a learning rate of $1 \times 10^{-6}$ for the optimizer, which can be further adjusted for performance optimization.
- Compile the model using Mean Squared Error (MSE) as the loss function and Stochastic Gradient Descent (SGD) as the optimizer.

```python
model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
)
```

**Training the Model**

- Train the model on the dataset for 100 epochs and evaluate the results.

```python
history = model.fit(dataset, epochs=100)
```

**Results**

- After training, the results shows better tracking of the original data, although the model still struggles with sharp increases.