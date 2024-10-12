# Questions and Answers

## 1. Question 1
What’s the primary difference between a simple RNN and an LSTM?

- [ ] In addition to the H output, RNNs have a cell state that runs across all cells.
- [ ] LSTMs have multiple outputs, RNNs have a single one.
- [x] In addition to the H output, LSTMs have a cell state that runs across all cells.
- [ ] LSTMs have a single output, RNNs have multiple.

**Correct**  
---

## 2. Question 2
If you want to clear out all temporary variables that TensorFlow might have from previous sessions, what code do you run?

- [ ] `tf.cache.clear_session()`
- [ ] `tf.cache.backend.clear_session()`
- [x] `tf.keras.backend.clear_session()`
- [ ] `tf.keras.clear_session`

**Correct**  
---

## 3. Question 3
What does a Lambda layer in a neural network do?

- [ ] Changes the shape of the input or output data.
- [ ] There are no Lambda layers in a neural network.
- [ ] Pauses training without a callback.
- [x] Allows you to execute arbitrary code while training.

**Correct**  
---

## 4. Question 4
If X is the standard notation for the input to an RNN, what are the standard notations for the outputs?

- [ ] Y
- [ ] H
- [x] Y(hat) and H
- [ ] H(hat) and Y

**Correct**  
---

## 5. Question 5
A new loss function was introduced in this module, named after a famous statistician. What is it called?

- [x] Huber loss
- [ ] Hyatt loss
- [ ] Hubble loss
- [ ] Hawking loss

**Correct**  
---

## 6. Question 6
What is a sequence to vector if an RNN has 30 cells numbered 0 to 29?

- [ ] The total Y(hat) for all cells.
- [ ] The Y(hat) for the second cell.
- [ ] The average Y(hat) for all 30 cells.
- [x] The Y(hat) for the last cell.

**Correct**  
---

## 7. Question 7
What does the axis parameter of tf.expand_dims do?

- [ ] Defines the axis around which to expand the dimensions.
- [x] Defines the dimension index at which you will expand the shape of the tensor.
- [ ] Defines if the tensor is X or Y.
- [ ] Defines the dimension index to remove when you expand the tensor.

**Correct**  
---

## 8. Question 8
What happens if you define a neural network with these three layers?

```python
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
tf.keras.layers.Dense(1)
```

- [ ] Your model will fail because you need return_sequences=True after each LSTM layer.
- [ ] Your model will compile and run correctly.
- [x] Your model will fail because you need return_sequences=True after the first LSTM layer.
- [ ] Your model will fail because you have the same number of cells in each LSTM.

**Correct**  
