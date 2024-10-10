# Time Series

- Time series are ordered sequences of values, typically equally spaced over time.
- Common examples include stock prices, weather forecasts, and historical trends.

**Types of Time Series**

- **Univariate Time Series**

A single value at each time step (e.g., daily weather temperature).
- **Multivariate Time Series**

Multiple values at each time step, representing related data points (e.g., CO2 concentratin 
vs. global temperature).

## Problems Solved by Time Series

Below are some of the key problems time series analysis helps solve:

**Forecasting or Prediction**

- Forecasting is the process of using historical data to predict future trends or values.
- Example: Machine learning can be used to analyze the birth and death rate chart for Japan, helping government agencies plan for societal impacts like retirement and immigration.

**Filling Missing Data (Imputation)**

- Imputation can also fill gaps in data, where some data points are missing or uncollected.
- Example: Machine learning can be used to fill gaps in the Moore's law chart, addressing missing chip release data from certain years through imputation techniques.

**Anomaly Detection**

- Time series predictions can be used to detect anomalies in datasets.
- Example: Machine learning can be used to analyze website logs to identify unusual spikes that may indicate potential denial-of-service (DoS) attacks.

**Pattern Recognition**

- Time series analysis can be used to uncover patterns in the data that reveal the source or cause of the series itself.
- Example: Machine learning can be used to analyze time series of sound waves to identify words or subwords, enabling speech recognition.

## Common Patterns in Time Series

Below are some of the common patterns observed in time series:

**Trend**

- A trend indicates a specific direction over time (e.g., upward or downward).
- Example: Moore's Law demonstrates an upward trend in technology performance over time.

**Seasonality**

- Seasonality refers to patterns that repeat at predictable intervals.
- Example: A chart showing active users on a software developers' website shows regular dips, suggesting a seasonal pattern where activity decreases on weekends when fewer users are working.

**Combination of Trend and Seasonality**

- Some time series may exhibit both a general trend and seasonal patterns.
- Example: A chart may show an overall upward trend with local peaks and troughs.

**Random Values and White Noise**

- Certain time series may contain random values without discernible trends or seasonality, often referred to as white noise.
- Example: A stock price series fluctuates randomly without any apparent pattern, showing a high degree of variability and unpredictability over time.

**Auto-correlated Time Series**

- Some series may lack trend and seasonality but display spikes at random time intervals.
- Example: A time series shows a value dependent on the previous value (e.g., 99% of the previous timestep) with occasional unpredictable spikes called innovations.

**Non-Stationary Time Series**

- Non-stationary series exhibit drastic changes over time.
- Example: A time series may show a positive trend and clear seasonality initially, but undergo a significant change (e.g., due to a financial crisis) resulting in a downward trend with no seasonality.

## Partitioning Methods

Below are the common partitioning methods used during the training of time series:

**Fixed Partitioning**

- **Creation of Sets:**
  - **Training Set:** The initial segment of the time series data, typically consisting of the first portion of data points (e.g., the first few years).
  - **Validation Set:** The subsequent segment immediately following the training set, used to tune model hyperparameters (e.g., the next year or two).
  - **Test Set:** The final segment, used to evaluate model performance after training and validation (e.g., the last year of data).
- **Characteristics:**
  - Each set must contain complete seasons to ensure all seasonal patterns are adequately represented.

**Roll Forward Partitioning**

- **Creation of Sets:**
  - **Initial Training Set:** A small initial segment of the time series (e.g., the first month or first few weeks).
  - **Validation Set:** The next immediate period following the training set (e.g., the day or week after the training set).
- **Subsequent Iterations:**
  - The training set is incrementally expanded by adding one more data point (e.g., one day or one week) from the series.
  - After each expansion, the model is trained on this new training set and used to forecast the next data point(s) in the validation set.
  - This process continues, rolling forward through the data, until the entire time series has been used for validation.

## Metrics for evaluating performance

To evaluate the performance of the model, you need to calculate the difference between forecasted values and actual values. The most common metrics are:

**Mean Squared Error (MSE)**

Common metric calculated by squaring errors and finding the mean to avoid cancellation of negative values.

```python
mse = np.square(errors).mean()
```

**Root Mean Squared Error (RMSE)**

Square root of MSE to bring the error back to the original scale.

```python
rmse = np.sqrt(mse)
```

**Mean Absolute Error (MAE)**

Uses absolute values of errors, penalizing large errors less than MSE.

```python
mae = np.abs(errors).mean()
```

**Mean Absolute Percentage Error (MAPE)**

Mean ratio of absolute error to actual values, indicating error size relative to values.

```python
mape = np.abs(errors / x_valid).mean()
```

## Moving average and differencing

Sure! Here’s a brief explanation of moving average and differencing using bullet points:

**Moving Average**

- Moving Average is a statistical method used to analyze data points by creating averages of different subsets of the full dataset.
- It is used to smoothens out short-term fluctuations and highlights longer-term trends or cycles.
- **Calculation**: 
  - An averaging window (e.g., 30 days) is defined.
  - The average of data points within this window is calculated at each point in time.
- **Advantages**:
  - Reduces noise in the data.
  - Provides a clearer view of the underlying trend.
- **Limitations**:
  - Does not account for trends or seasonality.
  - Can lag behind real-time changes in the data.

**Differencing**

- Differencing is a technique used to remove trends and seasonality from a time series by computing the difference between consecutive observations.
- It helps stabilize the mean of a time series by eliminating changes due to trends or seasonal effects.
- **Calculation**:
  - For each time point $ T $, the difference is calculated as:
     $$ \text{Difference} = x_T - x_{T-n} $$
     where:
     - $ x_T $ = value at time $ T $
     - $ x_{T-n} $ = value at an earlier time, where $ n $ is the specified period (e.g., 1 day, 365 days).
- **Advantages**:
  - Converts non-stationary data into stationary data, which is easier to model.
  - Allows for improved forecasting accuracy by focusing on changes rather than absolute values.
- **Limitations**:
  - May not fully eliminate seasonality if the seasonal patterns are complex.
  - The resulting series may lose information about the original data level.
