# Logistic Regression Overview

Logistic Regression is a statistical method for predicting binary outcomes based on some predictor variables. It is used when the dependent variable is categorical. The outcome is modeled using a logistic function, which ensures that the predictions are bounded between 0 and 1.

## Logistic Function

The logistic function, also known as the sigmoid function, is given by:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Where $ z = \theta^Tx $ and $ \theta $ is the parameter vector, $ x $ is the input feature vector, and $ e $ is the base of the natural logarithm.

## Cost Function

The cost function used in logistic regression, also known as the log loss, measures the performance of a classification model. For logistic regression, the cost function is defined as:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\sigma(\theta^Tx^{(i)})) + (1-y^{(i)}) \log(1-\sigma(\theta^Tx^{(i)}))] $$

where $ m $ is the number of training examples, $ y^{(i)} $ is the actual outcome, and $ \sigma(\theta^Tx^{(i)}) $ is the predicted probability.

## Gradient Descent

Gradient Descent is an optimization algorithm used for minimizing the cost function in various machine learning algorithms, including logistic regression. It involves iteratively moving towards the minimum of the cost function by updating the parameters in the opposite direction of the gradient of the cost function.

#### Update Rule

The parameters $ \theta $ are updated as follows:

$$ \theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j} $$

where $ \alpha $ is the learning rate, and $ \frac{\partial J(\theta)}{\partial \theta_j} $ is the partial derivative of the cost function with respect to the $ j $-th parameter.

## Practical Example with NumPy and Matplotlib

Now, let's move to a practical example where we will implement logistic regression and apply gradient descent to optimize the model parameters. We'll use NumPy for numerical computations and Matplotlib for visualization.

First, we need to load a dataset, preprocess it if necessary, and then we'll dive into implementing logistic regression and gradient descent.

## Step 1: Setup and Data Preparation

```python
import numpy as np
import matplotlib.pyplot as plt

# Example dataset loading and preparation steps
# Assume X and y are loaded, where X is the feature matrix and y is the label vector
```

## Step 2: Implementing Logistic Regression

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    predictions = sigmoid(np.dot(X, theta))
    cost = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        gradient = (1/m) * np.dot(X.T, (sigmoid(np.dot(X, theta)) - y))
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
        
    return theta, cost_history

```

## Training the Model

```python
# Initializations
theta_initial = np.zeros(X.shape[1])
alpha = 0.01
iterations = 1000

# Training
theta_optimal, cost_history = gradient_descent(X, y, theta_initial, alpha, iterations)

# Plotting the cost function
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Over Iterations')
plt.show()

```


