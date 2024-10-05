# Traditional Programming vs. Machine Learning

- **Traditional programming:** Developers write rules and use data to get answers.
  - Useful when problems are well-defined and rules can be easily coded, providing full control over the logic.
- **Machine learning:** Instead of coding rules, developers provide data and answers, and the machine learns the rules.
  - Useful for tasks where defining rules is complex, such as pattern recognition or natural language processing

**Machine Learning Shift**

- In this new approach, the machine learns rules from labeled data examples.
- Algorithms detect patterns, enabling solutions previously impractical with manually defined rules.

**Neural Networks and Deep Learning**

- Neural networks excel at pattern recognition.
- Deep learning, a more advanced form of machine learning, simplifies creating neural networks.
- It can be implemented with minimal code despite its complexity.

# The ‘Hello World’ of Neural Networks

  - Given a set of numbers (X and Y), as follow, can you infer the formula that maps X to Y?
     $$X = [-1, 0, 1, 2, 3, 4]$$
     $$Y = [-3, -1, 1, 3, 5, 7]$$
  - Humans can infer patterns like this by observing how Y changes as X changes, and refining guesses based on the data.
  - This is similar to how a machine learning model learns: by analyzing examples and adjusting to better match the pattern.
  - For your information, the formula that maps X to Y is $𝑌 = 2𝑋 − 1$.

## Neural Networks Basics

- A neural network consists of layers of neurons that learn patterns in data.
- Below, we use **Keras** (a high-level API in **TensorFlow**) to create a simple neural network with just one neuron.

```python
import tensorflow as tf
import numpy as np

# Define the simplest neural network with one neuron
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

# Compile the model with a loss function and optimizer
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training data: X and Y values following the pattern Y = 2X - 1
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(X, Y, epochs=500)
```

- The neural network has a single neuron (`Dense` layer) that learns the relationship between X and Y.
- The **loss function** (`mean_squared_error`) measures how far the model’s predictions are from the actual values.
- The **optimizer** (`sgd` for Stochastic Gradient Descent) adjusts the model’s parameters to minimize the loss and improve predictions.
- The model trains over **epochs** (500 iterations here), making predictions, calculating errors, and updating its guesses.

To make predictions with the trained model, use the following code:

```python
# Predict the value for X = 10
print(model.predict([10.0]))
```

- The model predicts a value close to 19 for \( X = 10 \), as \( Y = 2(10) - 1 = 19 \).
- Neural networks generalize from limited data, so predictions might not be exact but will be close to the expected result.