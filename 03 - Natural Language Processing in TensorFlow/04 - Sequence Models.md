# Sequence models

- A neural network can be seen as a function that, when provided with data and labels, infers rules from them.
- These inferred rules can later be used for making predictions or classifications.
- However, traditional neural networks do not inherently account for the sequence of data, which is critical for tasks like text analysis.

**Example of Sequence Importance: Fibonacci Sequence**

- Consider the Fibonacci sequence, where each number is the sum of the two preceding numbers:
    $$ 1, 1, 2, 3, 5, 8, 13 \dots $$
- This can be written as:
    $$ n_x = n_{x-1} + n_{x-2} $$
- For example, $3 = 2 + 1$, $5 = 2 + 3$, $8 = 3 + 5$, etc.
- This sequence demonstrates how each new value depends on previous values, which is an important concept for understanding sequential data.
- This idea of carrying information forward is foundational for Recurrent Neural Networks (RNNs).

**Introduction to Recurrent Neural Networks (RNNs)**

- An RNN handles data sequences by passing information between steps.
- Unlike traditional neural networks, RNNs retain information from previous steps to model sequences effectively.
- Basic RNN structure:
    - The input $x_0$ is fed into a function, producing an output $y_0$.
    - The output $y_0$ is passed into the next function along with the next input $x_1$, generating $y_1$.
    - This process continues, creating a chain where each output is influenced by both the current input and the previous output.
- This chaining effect allows RNNs to retain information from earlier inputs as they process the sequence.

## Long Short-Term Memory Networks (LSTMs)

- In basic text classification, context is usually derived from nearby words.
- However, in more complex sentences, important context words may appear much earlier, making accurate predictions more difficult.
- For example, in the sentence "I lived in Ireland, at school, they made me learn how to speak…", you might initially think of "Irish."
- However, "Irish" describes the people of Ireland, while the more accurate prediction would be "Gaelic," which is the language spoken there.
- The word that provides the necessary context, "Ireland," appears much earlier in the sentence.
- This can be problematic for standard RNNs, as they struggle with long-distance dependencies and primarily focus on passing immediate context from one step to the next.

**Introduction to LSTMs (Long Short-Term Memory Networks)**

- To solve this issue, LSTMs were developed as an improvement over RNNs.
- LSTMs include an additional **cell state pipeline** that helps retain important context over longer distances in a sequence.
- This allows context from earlier words (e.g., "Ireland") to influence predictions at later stages (e.g., "Gaelic").
- LSTMs can also be bidirectional, meaning that context from later words can influence earlier ones, further improving prediction accuracy.

## Implementing LSTMs in code

- To implement LSTMs in TensorFlow, use the `tf.keras.layers.LSTM` function.
- The primary parameter for LSTM layers is the number of outputs or units (e.g., 64 in this case).

**Bi-Directional LSTMs**

- To make the LSTM bidirectional, wrap it with `tf.keras.layers.Bidirectional`.
- This enables the LSTM to process input sequences in both directions, effectively doubling the output size.
- For example, if an LSTM has 64 units, the bi-directional LSTM will output 128 units.
    
```python
import tensorflow as tf

model = models.Sequential([
    tf.keras.Input(shape=(None,))
    tf.keras.layers.Embedding(VOCAB_SIZE, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

**Stacking LSTMs**

- You can stack multiple LSTM layers like any other Keras layer, but you need to set the parameter `return_sequences=True` for all LSTMs except the last one.
- This ensures that the output of the first LSTM matches the input format expected by the second LSTM.
  
```python
import tensorflow as tf

model = models.Sequential([
    tf.keras.Input(shape=(None,))
    tf.keras.layers.Embedding(VOCAB_SIZE, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), return_sequences=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Using LSTMs with Subword Tokens

- LSTMs are highly effective in models that use subword tokens, as they help retain context across sequences.
- This helps address the limitations found in earlier models, particularly when handling fragmented words due to tokenization.
- In the previous lab, we encountered issues when using subword tokenization in training a DNN.
- By applying LSTMs, we could achieve better results, especially with proper network tuning and hyperparameter adjustments.

**Accuracy Comparison Between One-Layer and Two-Layer LSTMs**

- When comparing the accuracy of a one-layer LSTM to a two-layer LSTM over **1/10 epochs**, the differences are minor.
- One notable issue is the **"nose dive" in validation accuracy** seen in the single-layer LSTM.
- The **training curve of the two-layer LSTM** is smoother, indicating better stability.
- **Jaggedness** in the training curve of the single-layer LSTM may suggest that the model needs improvement.

**Validation Accuracy**

- Both models reach a validation accuracy of around **80%**, despite challenges such as:
    - The training and test sets each contain **25,000 reviews**.
    - Only **8,000 subword tokens** were used from the training set, resulting in many **out-of-vocabulary (OOV)** tokens in the test set.
- Achieving **80% accuracy** is still considered good given the OOV issue.

**Loss Curve Analysis**

- The loss curve of the two-layer LSTM is smoother compared to the single-layer model, though it continues to **increase across epochs**.
- It’s recommended to monitor the loss in future epochs to ensure it **levels off**, which is desirable for better model performance.