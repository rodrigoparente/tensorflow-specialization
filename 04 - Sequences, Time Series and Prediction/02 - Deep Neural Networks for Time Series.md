# Deep Neural Networks for Time Series

- Last week, a synthetic seasonal dataset with trend, seasonality, and noise was created and analyzed using statistical methods.
- While results were promising, no machine learning (ML) was used.
- This week, ML techniques will be applied to enhance predictions on the same dataset.

## Preparing features and labels
  
- As with any ML problem, the dataset needs to be divided into features and labels.
- In this time series case, a sequence of values will be used as the feature, and the next value will be used as the label.
- The size of the sequence (number of values) used as the feature is called the **window size**.
- For example, using a window size of 30 means using 30 previous values as the feature to predict the 31st value (the label).

**Creating a Simple Dataset with TensorFlow**

To demonstrate this concept, the `tf.data.Dataset` class is used to create a dataset of 10 values (0 to 9).

```python
dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())
```

**Windowing the Dataset**

The dataset is expanded using windowing, with a window size of 5 and a shift of 1. This means the first window contains values `[0, 1, 2, 3, 4]`, the second window contains `[1, 2, 3, 4, 5]`, and so on.

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()
```

**Truncating Uneven Data**

To ensure that all windows are the same size, the `drop_remainder=True` parameter is used. This will truncate the remaining values that don't form a full window.
  
```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
    print(window.numpy())
```

**Splitting Features and Labels**

The data is split into features (all values except the last one) and labels (the last value).
  
```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
for x, y in dataset:
    print(x.numpy(), y.numpy())
```

**Shuffling the Dataset**

Before training a machine learning model, it’s common practice to shuffle the dataset. This can be done with the `shuffle()` method by specifying a buffer size of 10.
  
```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
dataset = dataset.shuffle(buffer_size=10)
```

**Batching the Data**

The dataset is batched into groups of two, using the `batch()` method. Batching helps in efficiently feeding data into the model during training.
  
```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
```

## Feeding windowed dataset into neural network

- We can adapt the previous code to create a windowed dataset that will be fed to a neural network.
- The following Python function prepares the time series dataset for training by applying windowing, shuffling, batching, and mapping it to features and labels.

```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
```  

## Single layer neural network

- Let's start with a super simple one that's effectively a linear regression.
- Next, we'll measure its accuracy, and then we'll work from there to improve that.

**Splitting the dataset**

Before training, the dataset is split into training and validation sets. The following code sets the split point at time step 1,000:
  
```python
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```

**Setting up constants**

Constants such as window size, batch size, and shuffle buffer size are defined. These parameters are passed to the `windowed_dataset` function to format the dataset for training:
  
```python
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
```

**Creating the dataset**

Using the `windowed_dataset` function with the defined parameters, the time series data is formatted into the proper shape for model training:
  
```python
dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)
```

**Defining the model**

A simple neural network is set up with a single dense layer, essentially performing linear regression. The dense layer is stored in the variable `l0` for later reference:
  
```python
l0 = tf.keras.layers.Dense(1)
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,)),
    l0,
])
```

**Compiling the model**

The model is compiled with the Mean Squared Error (MSE) loss function and Stochastic Gradient Descent (SGD) optimizer. The learning rate and momentum parameters are set for fine-tuning:
    
```python
model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
)
```

**Training the model**

The model is trained for 100 epochs using the formatted dataset:

```python
model.fit(dataset, epochs=100)
```

**Inspecting the learned weights**

After training, the learned weights of the dense layer are printed. The first array contains 20 values, representing the weights for the input features, and the second array contains the bias (b) value:
  
```python
print("Layer weights {}".format(l0.get_weights()))
```

## Prediction

- After training the model, the learned weights (`w`) and bias (`b`) can be used to perform standard linear regression.
- The model predicts `y` at any step by multiplying the input values (`x`) by the learned weights and adding the bias.
  
**Making a Prediction**

We can pass a slice of 20 items from the time series into the model to get a prediction. The `np.newaxis` method reshapes the data to match the input dimensions expected by the model. 

```python
print(series[1:21])
model.predict(series[1:21][np.newaxis])
```

Output:
```plaintext
# 20 input values from the series:
[49.35275  53.314735 57.711823 48.934444 48.931244 57.982895 53.897125
47.67393  52.68371  47.591717 47.506374 50.959415 40.086178 40.919415
46.612473 44.228207 50.720642 44.454983 41.76799  55.980938]

# Predicted value:
array([[49.08478]], dtype=float32)
```

**Generating a forecast for the time series**

To predict future values, we iterate over the time series in steps of the window size (in this case, 20), generating predictions for each step. The predictions are stored in a list called `forecast`.

```python
forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
```

**Post-processing the forecast**

After generating the predictions, we extract the forecasted values starting from the split time to focus on the validation data. These are converted into a NumPy array for further analysis or plotting:

```python
forecast = forecast[split_time-window_size:]
results = np.array(forecast).squeeze()
```

**Evaluating performance**

Finally, the Mean Absolute Error (MAE) is calculated to assess the performance of the model, which is comparable to the results from earlier complex analyses.

```python
tf.keras.metrics.mae(x_valid, results).numpy()
```

## Deep Neural Network Training, Tuning and Prediction

- The DNN has three layers: two with 10 neurons (ReLU activation) and one with 1 neuron.  
- The input shape matches the window size, and the model is compiled with MSE loss and SGD optimizer (momentum 0.9).  
- It is trained for 100 epochs.

```python
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100)
```

- The resulting loss was low, but there is still room for improvement.

**Using a Learning Rate Scheduler for Optimization**

- Selecting the optimal learning rate enhances model performance.  
- A learning rate scheduler callback dynamically adjusts the learning rate during training.  
- This callback modifies the learning rate at the end of each epoch.
- The scheduler function starts at `1e-8` and increases the learning rate based on the epoch number.  

```python
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20)
)

history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
```

**Visualizing the Learning Rate and Loss**

- After training, plot the learning rate against the loss per epoch to identify the optimal rate.  
- The chart displays loss on the y-axis and learning rate on the x-axis, aiming to find the point with the lowest, stable loss.  
- The optimal learning rate is approximately `7 * 10^-6`.

```python
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
```

- Although the loss decreases after 500 epochs, earlier losses are much higher, skewing the overall chart.  
- Cropping early epochs highlights the model's performance improvement over time.  
- The results indicate a significantly lower Mean Absolute Error (MAE) compared to earlier models.