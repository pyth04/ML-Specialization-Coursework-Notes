# Gradient Descent for Linear Regression

Gradient descent is an essential optimization algorithm in machine learning, used to minimize the cost function, particularly in linear regression models. It helps in finding the parameters of the model that significantly reduce the prediction error.

## Mean Squared Error (MSE)
The Mean Squared Error (MSE) is a common cost function used in linear regression. It is defined as the average of the squared differences between the actual and predicted values:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

where:
- $`n`$ represents the number of observations.
- $`y_i`$ is the actual value for the ith observation.
- $`\hat{y}_i`$ is the predicted value for the ith observation, calculated as $`\hat{y}_i = wx_i + b`$.

## Parameters of the Linear Regression Model
In linear regression, the model makes predictions using a linear function of the input features:

$$
\hat{y} = wx + b
$$

where:
- $`w`$ represents the weight or coefficient (slope of the line).
- $`b`$ represents the bias or intercept (where the line crosses the y-axis).

## Deriving the Gradients for MSE

In a linear regression model, the Mean Squared Error (MSE) is used as a cost function to measure the performance of the model. The MSE is defined as the average of the squared differences between the actual values and the predicted values. The predicted value $\hat{y}_i$ for the ith observation is given by the linear equation $\hat{y}_i = wx_i + b$. 

### Gradient with respect to `w`

The gradient of MSE with respect to `w` is the partial derivative of the MSE with respect to `w`. Starting with the MSE formula:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

Substituting $\hat{y}_i = wx_i + b$ into the MSE formula:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (wx_i + b - y_i)^2
$$

To find the gradient, we differentiate MSE with respect to `w`:

$$
\frac{\partial MSE}{\partial w} = \frac{\partial}{\partial w} \left( \frac{1}{n} \sum_{i=1}^{n} (wx_i + b - y_i)^2 \right)
$$

Applying the chain rule of differentiation, we get:

$$
\frac{\partial MSE}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} x_i(wx_i + b - y_i)
$$

### Gradient with respect to `b`

Similarly, the gradient of MSE with respect to `b` is the partial derivative of the MSE with respect to `b`. Differentiating MSE with respect to `b`:

$$
\frac{\partial MSE}{\partial b} = \frac{\partial}{\partial b} \left( \frac{1}{n} \sum_{i=1}^{n} (wx_i + b - y_i)^2 \right)
$$

Applying the chain rule of differentiation, we obtain:

$$
\frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (wx_i + b - y_i)
$$

These gradients indicate how much the cost function (MSE) changes with a small change in `w` and `b`, respectively. They are used in the gradient descent algorithm to update `w` and `b` in the direction that minimally reduces MSE.

## Implementing Gradient Descent
The gradient descent algorithm updates the parameters `w` and `b` iteratively as follows:

### 1. Initialize `w` and `b`
      Start with initial guesses (often zeros).

### 2. Compute the Gradient
      Calculate the gradients of MSE concerning both w and b.

### 3. Update Parameters
      Adjust w and b using the update rules:

$$
w := w - \alpha \left( \frac{2}{n} \sum_{i=1}^{n} x_i(wx_i + b - y_i) \right)
$$
   
$$
b := b - \alpha \left( \frac{2}{n} \sum_{i=1}^{n} (wx_i + b - y_i) \right)
$$

where $`\alpha`$ is the learning rate.

### 4. Repeat
      Iterate steps 2 and 3 until the changes in w and b are minimal or a pre-defined number of iterations is reached.

## Convergence to the Optimal Solution

The convergence of the gradient descent algorithm to the optimal solution in a linear regression model is significantly influenced by the nature of the cost function. For linear regression, the Mean Squared Error (MSE) is typically used as the cost function, which has some crucial properties:

- **Convexity**:

   > The MSE cost function is convex in the context of linear regression. This means that the function has a bowl-shaped curve, and any local minimum is also a global minimum. The importance of convexity in the cost function cannot be overstated as it ensures that the solution space contains no local minima, other than the global minimum.

- **Gradient Descent and Convex Functions**:

   > When a cost function is convex, the gradient descent algorithm can effectively navigate towards the minimum point. Since there are no local minima other than the global minimum, the algorithm is not at risk of getting "trapped" in a local minimum and failing to find the optimal solution.

- **Optimal Values of Parameters**

   > The global minimum of the MSE cost function corresponds to the values of `w` (weight) and `b` (bias) where the linear regression model best fits the data. At this point, the difference between the predicted values and the actual values is minimized over the entire dataset. Therefore, the convergence of gradient descent to this minimum ensures that we have found the most accurate linear relationship as modeled by our linear regression equation.

- **Empirical Verification**:

   > In practice, we can empirically verify the convergence of gradient descent by observing the decrease in MSE with each iteration. When the change in MSE between iterations falls below a certain threshold, or after a predefined number of iterations, we can be confident that the algorithm has converged.

- **Implication for Predictive Performance**

   > The optimal values of `w` and `b`, found at the global minimum, are the parameters for which our linear model has the best predictive performance on the training data. This optimal line is the one that best reduces the prediction error, making our model as accurate as possible given the linear assumption.

In summary, the convex nature of the MSE cost function in linear regression models is a key factor that allows gradient descent to reliably find the optimal parameters (`w` and `b`). This leads to a model that best fits the given data, providing the most accurate predictions within the framework of linear regression.
