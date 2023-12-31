### Gradient Descent for Linear Regression

Gradient descent is an optimization algorithm used to minimize the cost function in linear regression models. It is particularly useful for finding the best-fit parameters that reduce the prediction error.

#### Cost Function in Linear Regression
The cost function, often chosen as the Mean Squared Error (MSE), measures the average of the squared differences between the predicted and actual values. It's expressed as:

\[
MSE = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
\]

where:
- `n` is the number of observations
- `y_i` is the actual value for the ith observation
- `\hat{y}_i` is the predicted value for the ith observation, given by `\hat{y}_i = wx_i + b`

#### Parameters of the Model
The linear regression model predicts values using the linear equation:

\[
\hat{y} = wx + b
\]

- `w` represents the weight or slope of the line
- `b` represents the bias or y-intercept of the line

#### Implementing Gradient Descent
Gradient descent minimizes the MSE by iteratively updating the parameters `w` and `b`:

1. **Initialize `w` and `b`**: Start with random values or zeros.
2. **Compute the Gradient**: Calculate the partial derivatives of the MSE with respect to `w` and `b`.
3. **Update Parameters**: Adjust `w` and `b` by subtracting a fraction of the gradient:

   \[
   w := w - \alpha \frac{\partial MSE}{\partial w}
   \]

   \[
   b := b - \alpha \frac{\partial MSE}{\partial b}
   \]

   where:
    - `:=` indicates an update
    - `\alpha` is the learning rate
    - `\frac{\partial MSE}{\partial w}` and `\frac{\partial MSE}{\partial b}` are the gradients with respect to `w` and `b`

4. **Iterate**: Repeat the gradient computation and parameter update until convergence.

The MSE cost function is convex, guaranteeing that gradient descent can converge to the global minimum, resulting in the most accurate predictions for a linear relationship.
