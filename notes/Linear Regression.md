#### Linear Regression Model
The linear regression model predicts the value of a dependent variable `y` based on the independent variable `x`. The model is defined by the parameters `w` (weight) and `b` (bias):

$$ y = wx + b $$

- `w` represents the weight or slope of the line.
- `b` represents the bias or y-intercept of the line.
- `x` is the independent variable.
- `y` is the predicted dependent variable.

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

#### Gradient Descent Algorithm
Gradient descent is used to find the values of `w` and `b` that minimize the cost function.

##### For MSE
- Gradient w.r.t. `w`:
  $$ \frac{\partial \text{MSE}}{\partial w} = \frac{-2}{n} \sum_{i=1}^{n} x_i(y_i - (wx_i + b)) $$
- Gradient w.r.t. `b`:
  $$ \frac{\partial \text{MSE}}{\partial b} = \frac{-2}{n} \sum_{i=1}^{n} (y_i - (wx_i + b)) $$

##### For RMSE
- Gradient w.r.t. `w`:
  $$ \frac{\partial \text{RMSE}}{\partial w} = \frac{-1}{n \cdot \text{RMSE}} \sum_{i=1}^{n} x_i(y_i - (wx_i + b)) $$
- Gradient w.r.t. `b`:
  $$ \frac{\partial \text{RMSE}}{\partial b} = \frac{-1}{n \cdot \text{RMSE}} \sum_{i=1}^{n} (y_i - (wx_i + b)) $$

##### For MAE
- Gradient w.r.t. `w`:
  $$ \frac{\partial \text{MAE}}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} x_i \cdot \text{sign}(y_i - (wx_i + b)) $$
- Gradient w.r.t. `b`:
  $$ \frac{\partial \text{MAE}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} \text{sign}(y_i - (wx_i + b)) $$

