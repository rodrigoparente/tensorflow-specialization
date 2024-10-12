# Questions and Answers

## 1. Question 1
You used a sunspots dataset that was stored in CSV. What’s the name of the Python library used to read CSVs?

- [ ] py_files
- [ ] CommaSeparatedValues
- [x] csv
- [ ] py_csv

**Correct**  
---

## 2. Question 2
Why is MAE a good analytic for measuring accuracy of predictions for time series?

- [x] It doesn’t heavily punish larger errors like square errors do.
- [ ] It biases towards small errors.
- [ ] It punishes larger errors.
- [ ] It only counts positive errors.

**Correct**  
---

## 3. Question 3
What is the expected input shape for a univariate time series to a Conv1D?

- [ ] (1,)
- [ ] (1, window_size)
- [x] (window_size, 1)
- [ ] ()

**Correct**  
---

## 4. Question 4
When you read a row from a reader and want to cast column 2 to another data type, for example, a float, what’s the correct syntax?

- [ ] `float f = row[2].read()`
- [ ] You can’t. It needs to be read into a buffer and a new float instantiated from the buffer.
- [ ] `Convert.toFloat(row[2])`
- [x] `float(row[2])`

**Correct**  
---

## 5. Question 5
How do you add a 1-dimensional convolution to your model for predicting time series data?

- [ ] Use a 1DConvolution layer type.
- [ ] Use a 1DConv layer type.
- [x] Use a Conv1D layer type.
- [ ] Use a ConvolutionD1 layer type.

**Correct**  
---

## 6. Question 6
If your CSV file has a header that you don’t want to read into your dataset, what do you execute before iterating through the file using a ‘reader’ object?

- [x] `next(reader)`
- [ ] `reader.read(next)`
- [ ] `reader.next`
- [ ] `reader.ignore_header()`

**Correct**  
---

## 7. Question 7
After studying this course, what neural network type do you think is best for predicting time series like our sunspots dataset?

- [ ] Convolutions
- [ ] RNN / LSTM
- [x] A combination of all other answers
- [ ] DNN

**Correct**
