#### Cost Functions
##### 1. Mean Squared Error (MSE)
MSE measures the average squared difference between the actual and predicted values.

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2 $$

- `y_i` is the actual value for the `i`th data point.
- `wx_i + b` is the predicted value for the `i`th data point.
- `n` is the number of data points.

##### 2. Root Mean Squared Error (RMSE)
RMSE is the square root of MSE, providing error in the same units as the output.

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2} $$

##### 3. Mean Absolute Error (MAE)
MAE measures the average absolute difference between the actual and predicted values.

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - (wx_i + b)| $$

