# Questions and Answers

## 1. Question 1
What does MAE stand for?

- [ ] Mean Average Error
- [ ] Mean Advanced Error
- [x] Mean Absolute Error
- [ ] Mean Active Error

**Correct**  
---

## 2. Question 2
What’s the correct line of code to split an n column window into n-1 columns for features and 1 column for a label?

- [ ] `dataset = dataset.map(lambda window: (window[n-1], window[1]))`
- [x] `dataset = dataset.map(lambda window: (window[:-1], window[-1:]))`
- [ ] `dataset = dataset.map(lambda window: (window[-1:], window[:-1]))`
- [ ] `dataset = dataset.map(lambda window: (window[n], window[1]))`

**Correct**  
---

## 3. Question 3
If you want to inspect the learned parameters in a layer after training, what’s a good technique to use?

- [ ] Iterate through the layers dataset of the model to find the layer you want.
- [x] Assign a variable to the layer and add it to the model using that variable. Inspect its properties after training.
- [ ] Decompile the model and inspect the parameter set for that layer.
- [ ] Run the model with unit data and inspect the output for that layer.

**Correct**  
---

## 4. Question 4
If you want to amend the learning rate of the optimizer on the fly, after each epoch. What do you do?

- [ ] Use a LearningRateScheduler object and assign that to the `callbacks` parameter in model.compile()
- [ ] Callback to a custom function and change the SGD property
- [x] Use a LearningRateScheduler object in the callbacks namespace and assign that to the `callbacks` parameter in model.fit()
- [ ] You can’t set it

**Correct**  
---

## 5. Question 5
What does ‘drop_remainder=True’ do?

- [x] It ensures that all rows in the data window are the same length by cropping data
- [ ] It ensures that the data is all the same shape
- [ ] It ensures that all data is used
- [ ] It ensures that all rows in the data window are the same length by adding data

**Correct**  
---

## 6. Question 6
What does MSE stand for?

- [ ] Mean Series error
- [ ] Mean Slight error
- [ ] Mean Second error
- [x] Mean Squared error

**Correct**  
---

## 7. Question 7
If time values are in time[], series values are in series[] and we want to split the series into training and validation at time split_time, what is the correct code?

- [ ] 
```python
time_train = time[split_time]
x_train = series[split_time]
time_valid = time[split_time]
x_valid = series[split_time]
```

- [ ] 
```python
time_train = time[split_time]
x_train = series[split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```

- [ ] 
```python
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time]
x_valid = series[split_time]
```

- [x] 
```python
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```

**Correct**  
---

## 8. Question 8
How do you set the learning rate of the SGD optimizer?

- [ ] Use the RateOfLearning property
- [ ] Use the Rate property 
- [x] Use the learning_rate property
- [ ] You can’t set it

**Correct**  
---

## 9. Question 9
What is a windowed dataset?

- [ ] A consistent set of subsets of a time series
- [ ] There’s no such thing
- [ ] The time series aligned to a fixed shape
- [x] A fixed-size subset of a time series 

**Correct**
