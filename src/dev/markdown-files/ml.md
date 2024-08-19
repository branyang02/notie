# **Machine Learning**

<span class="subtitle">
Fall 2024 | Author: Brandon Yang
</span>

## Introduction

Machine learning is a field of artificial intelligence that focuses on developing algorithms and models that can learn from data to make predictions and decisions. Machine learning algorithms can be broadly categorized into three types:

1. **Supervised Learning**: Algorithms learn from labeled data to predict outcomes or recognize patterns.
2. **Unsupervised Learning**: Algorithms learn from unlabeled data to discover hidden patterns or structures.
3. **Reinforcement Learning**: Algorithms learn through trial and error to maximize rewards in a given environment.

Every machine learning model consists of two main components:

1. **Model**: A mathematical representation of the relationship between the input features and the target values.
2. **Learning Algorithm**: A method to optimize the model parameters based on the training data.

A model is essentially a mapping function that takes input features and produces output predictions. Let's call this function the **hypothesis function** $h(\cdot)$:

$$
h: \mathcal{X} \rightarrow \mathcal{Y},
$$

where $\mathcal{X}$ is the input space (features) and $\mathcal{Y}$ is the output space (target values). Since the goal of machine learning is to find the optimal **model parameters**, we can to define the hypothesis function as:

$$
\hat{\mathbf{y}} = h_{\theta}(\mathbf{x}) = f(\mathbf{x}; \theta),
$$

where $\theta$ is the **parameter** that we aim to optimize, and $f(\cdot)$ is the model function that maps the input features $\mathbf{x} \in \mathcal{X}$ to the predicted values $\hat{\mathbf{y}} \in \mathcal{Y}$. Finding the optimal $\theta$ is the same as minimizing some **loss function** that measures the difference between the predicted values and the actual values. Therefore, the learning algorithm aims to minimize the loss function typically defined as $L(\theta)$. There are various optimization algorithms to minimize the loss function and find the optimal parameter vector $\theta$.

To summarize, for every machine learning model, we need to define the following components:

1. **Model**: A hypothesis function $h_{\theta}(\mathbf{x}) = f(\mathbf{x}; \theta)$ that maps input features to predicted values.
2. **Loss Function**: A function $L(\theta)$ that measures the difference between the predicted values and the actual values.

Of course, we can't forget about the data!

## Regression & Gradient Descent

In this section we will cover the following topics:

- [Linear Regression](#linear-regression)
- [Normal Equation](#normal-equation)
- [Gradient Descent](#gradient-descent)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)

### Linear Regression

Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable and one or more independent variables. The goal of linear regression is to find the best-fitting linear relationship between the input features and the target values.

<blockquote class="definition">

Supervised learning is a category of machine learning that uses **labeled** data to train algorithms to predict outcomes and recognize patterns.

</blockquote>

#### Data Representation

Given a set of observations $\mathbf{X} = \begin{bmatrix}
\mathbf{x}^{(1)} \\
\mathbf{x}^{(2)} \\
\vdots \\
\mathbf{x}^{(m)}
\end{bmatrix}$, where each $\mathbf{x}^{(i)}$ is a feature vector, and a set of corresponding target values (aka. labels) $\mathbf{y} = \begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)}
\end{bmatrix}$, we aim to find a parameter vector $\mathbf{\theta}$ that minimizes the difference between the predicted values and the actual values.

To compute the predicted values given the parameter vector $\mathbf{\theta}$, we define the hypothesis function as:

$$
\hat{y} = h_{\mathbf{\theta}}(\mathbf{x}) = \mathbf{\theta}^T \mathbf{x} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n,
$$

where $\mathbf{x}$ is the feature vector with the first element set to 1, and $\hat{y}$ is the predicted label. We set the first element to 1 to account for the bias term $\theta_0$ in the hypothesis function.

We do this to simplify the notation and allow the bias term $\theta_0$ to be included in the parameter vector $\mathbf{\theta}$. Therefore, the parameter vector $\mathbf{\theta}$ is of size $n+1$. Next, we can represent the hypothesis function in matrix form as:

$$
\hat{\mathbf{y}} = h_{\mathbf{\theta}}(\mathbf{X}) = \mathbf{X}\mathbf{\theta}, \text{ where } \mathbf{X} = \begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{bmatrix},
$$

where $\hat{\mathbf{y}} = \begin{bmatrix}
\hat{y}^{(1)} \\
\hat{y}^{(2)} \\
\vdots \\
\hat{y}^{(m)}
\end{bmatrix}$ is the predicted label vector.

To perform optimization on the parameter vector $\mathbf{\theta}$, we need to define a loss function that measures the difference between the predicted values and the actual values.

#### Loss Funciton

One common loss function used in linear regression is the **mean squared error (MSE)**, which is defined as:

<blockquote class="equation">

**Mean Squared Error (MSE)**:

$$
\begin{equation} \label{eq:mse}
\begin{split}
\text{MSE}(\mathbf{\theta}) &= \frac{1}{m} \sum_{i=1}^{m} \left( h_{\mathbf{\theta}}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 \\
&= \frac{1}{m} \left\| h_{\mathbf{\theta}}(\mathbf{X}) - \mathbf{y} \right\|_2^2
\end{split}
\end{equation}
$$

</blockquote>

where $m$ is the number of observations, $\mathbf{X}$ is the feature matrix, and $\mathbf{y}$ is the target vector. The goal is to minimize the loss function by finding the optimal parameter vector $\mathbf{\theta}$.

#### Normal Equation

The normal equation is a closed-form solution to the linear regression problem. It is defined as:

<blockquote class="equation">

**Normal Equation**:

$$
\begin{equation} \label{eq:normal-equation}
\mathbf{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\end{equation}
$$

</blockquote>

where $\mathbf{X}^T$ is the transpose of the feature matrix $\mathbf{X}$, and $\mathbf{X}^T \mathbf{X}$ is a square matrix. The normal equation provides an analytical solution to the linear regression problem.

<details><summary>Normal Equation Proof</summary>

<blockquote class="proof">

To prove the normal equation $\eqref{eq:normal-equation}$, we start by rearranging $\eqref{eq:mse}$:

$$
\begin{aligned}
\text{MSE}(\mathbf{\theta}) &= \frac{1}{m} \left\| \mathbf{X}\mathbf{\theta} - \mathbf{y} \right\|_2^2 \\
&= \frac{1}{m} (\mathbf{X}\mathbf{\theta} - \mathbf{y})^T (\mathbf{X}\mathbf{\theta} - \mathbf{y}) \\
&= \frac{1}{m} (\mathbf{\theta}^T \mathbf{X}^T - \mathbf{y}^T) (\mathbf{X}\mathbf{\theta} - \mathbf{y}) \\
&= \frac{1}{m} (\mathbf{\theta}^T \mathbf{X}^T \mathbf{X} \mathbf{\theta} - \mathbf{\theta}^T \mathbf{X}^T \mathbf{y} - \mathbf{y}^T \mathbf{X} \mathbf{\theta} + \mathbf{y}^T \mathbf{y}) \\
&= \frac{1}{m} (\mathbf{\theta}^T \mathbf{X}^T \mathbf{X} \mathbf{\theta} - 2\mathbf{\theta}^T \mathbf{X}^T \mathbf{y} + \mathbf{y}^T \mathbf{y})
\end{aligned}
$$

To find the optimal parameter vector $\mathbf{\theta}$, we take the derivative of the loss function with respect to $\mathbf{\theta}$ and set it to zero:

$$
\begin{aligned}
\frac{\partial \text{MSE}(\mathbf{\theta})}{\partial \mathbf{\theta}} &= \frac{2}{m} \mathbf{X}^T \mathbf{X} \mathbf{\theta} - \frac{2}{m} \mathbf{X}^T \mathbf{y} = 0 \\
\mathbf{X}^T \mathbf{X} \mathbf{\theta} &= \mathbf{X}^T \mathbf{y} \\
\mathbf{\theta} &= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\end{aligned}
$$

Therefore, the normal equation provides the optimal parameter vector $\mathbf{\theta}$ that minimizes the loss function.

</blockquote>

</details>

<details><summary>Normal Equation Example</summary>

We can implement the normal equation for linear regression in Python. In this example, we generate random data points and use the normal equation to find the optimal parameter vector $\mathbf{\theta}$. We plot the computed regression line and the data points.

```execute-python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
m = 100  # Number of examples
X = 2 * np.random.rand(m, 1)  # Features (100x1 matrix)
y = 4 + 3 * X + np.random.randn(m, 1)  # Target values with some noise

# Add bias term (column of 1s) to X
X_b = np.c_[np.ones((m, 1)), X]  # X_b = [1, X] (100x2 matrix)

# Normal Equation
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Final parameter vector (theta)
print("Theta (parameters):", theta)

# Plotting the results
plt.figure(figsize=(8, 4))
plt.plot(X, y, "b.")
plt.plot(X, X_b.dot(theta), "r-")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Normal Equation")

get_image(plt)
```

The normal equation provides the optimal parameter vector $\mathbf{\theta}$ that minimizes the loss function (MSE). The regression line fits the data points well, and the loss is minimized.

</details>

Although the normal equation provides an optimal solution to linear regression, it is computationally expensive for large datasets due to the matrix inversion operation. For large datasets, we use iterative optimization algorithms such as **gradient descent**.

### Gradient Descent

Gradient descent is an iterative optimization algorithm used to minimize the loss function by updating the parameter vector $\mathbf{\theta}$ in the direction of the negative gradient. The update rule for gradient descent is defined as:

<blockquote class="equation">

**Gradient Descent Update Rule**:

$$
\begin{equation} \label{eq:gd}
\mathbf{\theta}^{(t+1)} = \mathbf{\theta}^{(t)} - \alpha \nabla_{\mathbf{\theta}} L(\mathbf{\theta})
\end{equation}
$$

</blockquote>

where $L(\mathbf{\theta})$ is the loss function, and $\alpha$ is the learning rate that controls the step size of the update. In the case of linear regression with the MSE loss function, the gradient is given by:

$$
\begin{equation} \label{eq:mse-gradient}
\nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) = \frac{2}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}),
\end{equation}
$$

where $\hat{\mathbf{y}} = \mathbf{X}\mathbf{\theta}$ is the predicted label vector.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec nulla auctor, tincidunt nunc nec, ultricies nunc. Nullam nec nunc nec nunc. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec nulla auctor, tincidunt nunc nec, ultricies nunc. Nullam nec nunc nec nunc. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec nulla auctor, tincidunt nunc nec, ultricies nunc. Nullam nec nunc nec nunc. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec nulla auctor, tincidunt nunc nec, ultricies nunc. Nullam nec nunc nec nunc.

<blockquote class="proof">

To compute $\eqref{eq:mse-gradient}$, we start by expanding Equation $\eqref{eq:mse}$:

$$
\begin{aligned}
\text{MSE}(\mathbf{\theta}) &= \frac{1}{m} \left\| \mathbf{X}\mathbf{\theta} - \mathbf{y} \right\|_2^2 \\
&= \frac{1}{m} (\mathbf{X}\mathbf{\theta} - \mathbf{y})^T (\mathbf{X}\mathbf{\theta} - \mathbf{y}) \\
&= \frac{1}{m} (\mathbf{\theta}^T \mathbf{X}^T \mathbf{X} \mathbf{\theta} - 2\mathbf{\theta}^T \mathbf{X}^T \mathbf{y} + \mathbf{y}^T \mathbf{y})
\end{aligned}
$$

Next, we compute the gradient with respect to $\mathbf{\theta}$:

$$
\begin{aligned}
\nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) &= \nabla_{\mathbf{\theta}} \left( \frac{1}{m} (\mathbf{\theta}^T \mathbf{X}^T \mathbf{X} \mathbf{\theta} - 2\mathbf{\theta}^T \mathbf{X}^T \mathbf{y} + \mathbf{y}^T \mathbf{y}) \right) \\
\end{aligned}
$$

To compute the gradient, we use the following properties:

1. $\nabla_{\mathbf{x}} \mathbf{x}^T \mathbf{A} \mathbf{x} = 2 \mathbf{A} \mathbf{x}$, where $\mathbf{A}$ is a symmetric matrix.
2. $\nabla_{\mathbf{x}} \mathbf{a}^T \mathbf{x} = \mathbf{a}$, where $\mathbf{a}$ is a vector.
3. $A^TA$ is symmetric for any matrix $A$.

Applying these properties, we get:

$$
\begin{aligned}
\nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) &= \frac{1}{m} \left( 2 \mathbf{X}^T \mathbf{X} \mathbf{\theta} - 2 \mathbf{X}^T \mathbf{y} \right) \\
&= \frac{2}{m} \mathbf{X}^T (\mathbf{X}\mathbf{\theta} - \mathbf{y})
\end{aligned}
$$

Therefore, the gradient of the MSE loss function with respect to $\mathbf{\theta}$ is given by $\eqref{eq:mse-gradient}$.

</blockquote>

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec nulla auctor, tincidunt nunc nec, ultricies nunc. Nullam nec nunc nec nunc.

<details><summary>MSE Gradient Proof</summary>

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec nulla auctor, tincidunt nunc nec, ultricies nunc. Nullam nec nunc nec nunc.

<blockquote class="proof">

To compute $\eqref{eq:mse-gradient}$, we start by expanding Equation $\eqref{eq:mse}$:

$$
\begin{aligned}
\text{MSE}(\mathbf{\theta}) &= \frac{1}{m} \left\| \mathbf{X}\mathbf{\theta} - \mathbf{y} \right\|_2^2 \\
&= \frac{1}{m} (\mathbf{X}\mathbf{\theta} - \mathbf{y})^T (\mathbf{X}\mathbf{\theta} - \mathbf{y}) \\
&= \frac{1}{m} (\mathbf{\theta}^T \mathbf{X}^T \mathbf{X} \mathbf{\theta} - 2\mathbf{\theta}^T \mathbf{X}^T \mathbf{y} + \mathbf{y}^T \mathbf{y})
\end{aligned}
$$

Next, we compute the gradient with respect to $\mathbf{\theta}$:

$$
\begin{aligned}
\nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) &= \nabla_{\mathbf{\theta}} \left( \frac{1}{m} (\mathbf{\theta}^T \mathbf{X}^T \mathbf{X} \mathbf{\theta} - 2\mathbf{\theta}^T \mathbf{X}^T \mathbf{y} + \mathbf{y}^T \mathbf{y}) \right) \\
\end{aligned}
$$

To compute the gradient, we use the following properties:

1. $\nabla_{\mathbf{x}} \mathbf{x}^T \mathbf{A} \mathbf{x} = 2 \mathbf{A} \mathbf{x}$, where $\mathbf{A}$ is a symmetric matrix.
2. $\nabla_{\mathbf{x}} \mathbf{a}^T \mathbf{x} = \mathbf{a}$, where $\mathbf{a}$ is a vector.
3. $A^TA$ is symmetric for any matrix $A$.

Applying these properties, we get:

$$
\begin{aligned}
\nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) &= \frac{1}{m} \left( 2 \mathbf{X}^T \mathbf{X} \mathbf{\theta} - 2 \mathbf{X}^T \mathbf{y} \right) \\
&= \frac{2}{m} \mathbf{X}^T (\mathbf{X}\mathbf{\theta} - \mathbf{y})
\end{aligned}
$$

Therefore, the gradient of the MSE loss function with respect to $\mathbf{\theta}$ is given by $\eqref{eq:mse-gradient}$.

</blockquote>

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec nulla auctor, tincidunt nunc nec, ultricies nunc. Nullam nec nunc nec nunc.

</details>

The gradient descent algorithm for linear regression is as follows:

1. Initialize the parameter vector $\mathbf{\theta}$.
2. Compute the predicted values $\hat{\mathbf{y}} = \mathbf{X}\mathbf{\theta}$.
3. Compute the loss function $\text{MSE}(\mathbf{\theta}) = \frac{1}{m} \left\| \hat{\mathbf{y}} - \mathbf{y} \right\|_2^2$.
4. Compute the gradient of the loss function $\nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) = \frac{2}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$.
5. Update the parameter vector using the gradient descent update rule: $\mathbf{\theta} = \mathbf{\theta} - \alpha \nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta})$.
6. Repeat steps 2-5 until convergence.

<details><summary>Gradient Descent Example</summary>

We can implement the gradient descent algorithm for linear regression in Python. In this example, we also add a termination condition based on the change in the loss function to stop the optimization process when the loss converges. We plot both the computed regression line and the convergence of the loss function.

The termination condition is computed as:

$$
\text{abs}(L^{(t-1)} - L^{(t)}) < \text{tolerance},
$$

where $L^{(t)}$ is the loss at iteration $t$, and $\text{tolerance}$ is a small value to determine convergence.

```execute-python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
m = 100  # Number of examples
X = 2 * np.random.rand(m, 1)  # Features (100x1 matrix)
y = 4 + 3 * X + np.random.randn(m, 1)  # Target values with some noise

# Add bias term (column of 1s) to X
X_b = np.c_[np.ones((m, 1)), X]  # X_b = [1, X] (100x2 matrix)

# Initialize parameter vector theta (2x1 matrix)
theta = np.random.randn(2, 1)

# Learning rate
alpha = 0.1
# Maximum number of iterations
n_iterations = 10000000
# Tolerance for convergence
tolerance = 1e-10

# Function to compute MSE loss
def compute_mse(X_b, y, theta):
    m = len(y)
    predictions = X_b.dot(theta)
    mse = (1/m) * np.sum((predictions - y) ** 2)
    return mse

# Gradient Descent with termination condition
loss_values = []
for iteration in range(n_iterations):
    y_hat = X_b.dot(theta)
    loss = compute_mse(X_b, y, theta)
    gradients = 2/m * X_b.T.dot(y_hat - y)
    theta = theta - alpha * gradients
    loss_values.append(loss)

    # Termination condition
    if iteration > 0 and abs(loss_values[-2] - loss_values[-1]) < tolerance:
        print(f"Convergence reached at iteration {iteration}")
        break

# Final parameter vector (theta)
print("Theta (parameters):", theta)

# Final loss
print("Final Loss (MSE):", loss_values[-1])

# Plotting the results
plt.figure(figsize=(8, 4))

# Plot data and regression line
plt.subplot(1, 2, 1)
plt.plot(X, y, "b.")
plt.plot(X, X_b.dot(theta), "r-")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Gradient Descent")

# Plot loss values
plt.subplot(1, 2, 2)
plt.plot(loss_values, "g-")
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Loss Convergence")

plt.tight_layout()

get_image(plt)
```

As shown in the plot, the gradient descent algorithm converges to the optimal parameter vector $\mathbf{\theta}$ that minimizes the loss function (MSE). The regression line fits the data points well, and the loss decreases over iterations until convergence.

</details>

### Stochastic Gradient Descent

Stochastic gradient descent (SGD) is a variant of gradient descent that updates the parameter vector using a single training example at a time, instead of the entire dataset. This approach is computationally efficient for large datasets and can escape local minima due to the stochastic nature of the updates.

The update rule for stochastic gradient descent is defined as:

<blockquote class="equation">

**Stochastic Gradient Descent Update Rule**:

$$
\begin{equation} \label{eq:sgd}
\mathbf{\theta}^{(t+1)} = \mathbf{\theta}^{(t)} - \alpha \nabla_{\mathbf{\theta}} L(\mathbf{\theta}^{(t)})
\end{equation}
$$

</blockquote>

where $L(\mathbf{\theta}^{(t)})$ is the loss function computed using a single training example at iteration $t$. The gradient is computed with respect to the loss function of the current training example.

In the case of linear regression with the MSE loss function, the gradient for a single training example $(\mathbf{x}^{(i)}, y^{(i)})$ is given by:

$$
\begin{equation} \label{eq:mse-gradient-sgd}
\nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) = 2 \mathbf{x}^{(i)} (\hat{y}^{(i)} - y^{(i)}),
\end{equation}
$$

where $\hat{y}^{(i)} = \mathbf{\theta}^T \mathbf{x}^{(i)}$ is the predicted value for the training example.

<details><summary>MSE Gradient for SGD</summary>

To compute the gradient of the MSE loss function for stochastic gradient descent, we start by expanding the loss function for a single training example:

$$
\begin{aligned}
\text{MSE}^{(i)}(\mathbf{\theta}) &= \left( \mathbf{\theta}^T \mathbf{x}^{(i)} - y^{(i)} \right)^2 \\
&= \left( \mathbf{x}^{(i)T} \mathbf{\theta} - y^{(i)} \right)^2.
\end{aligned}
$$

To find the gradient with respect to $\mathbf{\theta}$, we take the derivative of the loss function:

$$
\nabla_{\mathbf{\theta}} \text{MSE}^{(i)}(\mathbf{\theta}) = \frac{\partial}{\partial \mathbf{\theta}} \left( \mathbf{x}^{(i)T} \mathbf{\theta} - y^{(i)} \right)^2.
$$

We can simply by setting $u = \mathbf{x}^{(i)T} \mathbf{\theta} - y^{(i)}$. Thus we have:

$$
\begin{aligned}
\nabla_{\mathbf{\theta}} \text{MSE}^{(i)}(\mathbf{\theta}) &= \frac{\partial}{\partial \mathbf{\theta}} u^2 \\
&= 2u \frac{\partial u}{\partial \mathbf{\theta}}.
\end{aligned}
$$

Now, we compute the derivative of $u$ with respect to $\mathbf{\theta}$:

$$
\begin{aligned}
\frac{\partial u}{\partial \mathbf{\theta}} &= \frac{\partial}{\partial \mathbf{\theta}} \left( \mathbf{x}^{(i)T} \mathbf{\theta} - y^{(i)} \right) \\
&= \mathbf{x}^{(i)}.
\end{aligned}
$$

Therefore, the gradient of the MSE loss function for a single training example is:

$$
\nabla_{\mathbf{\theta}} \text{MSE}^{(i)}(\mathbf{\theta}) = 2 \mathbf{x}^{(i)} (\hat{y}^{(i)} - y^{(i)}),
$$

where $\hat{y}^{(i)} = \mathbf{\theta}^T \mathbf{x}^{(i)}$ is the predicted value for the training example.

</details>

SGD for linear regression is as follows:

1. Initialize the parameter vector $\mathbf{\theta}$.
2. Shuffle the training data.
3. For each training example $(\mathbf{x}^{(i)}, y^{(i)})$:
   - Compute the predicted value $\hat{y}^{(i)} = \mathbf{\theta}^T \mathbf{x}^{(i)}$.
   - Compute the loss $L^{(i)} = (\hat{y}^{(i)} - y^{(i)})^2$.
   - Compute the gradient $\nabla_{\mathbf{\theta}} L^{(i)} = 2 \mathbf{x}^{(i)} (\hat{y}^{(i)} - y^{(i)})$.
   - Update the parameter vector $\mathbf{\theta} = \mathbf{\theta} - \alpha \nabla_{\mathbf{\theta}} L^{(i)}$.
4. Repeat steps 2-3 for multiple epochs.

<details><summary>Stochastic Gradient Descent Example</summary>

We can implement the stochastic gradient descent algorithm for linear regression in Python. In this example, we use a single training example at each iteration to update the parameter vector. We also plot the convergence of the loss function over epochs.

```execute-python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
m = 100  # Number of examples
X = 2 * np.random.rand(m, 1)  # Features (100x1 matrix)
y = 4 + 3 * X + np.random.randn(m, 1)  # Target values with some noise

# Add bias term (column of 1s) to X
X_b = np.c_[np.ones((m, 1)), X]  # X_b = [1, X] (100x2 matrix)

# Initialize parameter vector theta (2x1 matrix)
theta = np.random.randn(2, 1)

# Learning rate
alpha = 0.01
# Number of epochs
n_epochs = 50

# Function to compute MSE loss
def compute_mse(X_b, y, theta):
    m = len(y)
    predictions = X_b.dot(theta)
    mse = (1/m) * np.sum((predictions - y) ** 2)
    return mse

# Stochastic Gradient Descent
loss_values = []
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        y_hat = xi.dot(theta)
        loss = compute_mse(X_b, y, theta)
        gradients = 2 * xi.T.dot(y_hat - yi)
        theta = theta - alpha * gradients
        loss_values.append(loss)

# Final parameter vector (theta)
print("Theta (parameters):", theta)

# Final loss
print("Final Loss (MSE):", loss_values[-1])

# Plotting the results
plt.figure(figsize=(8, 4))

# Plot data and regression line
plt.subplot(1, 2, 1)
plt.plot(X, y, "b.")
plt.plot(X, X_b.dot(theta), "r-")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Stochastic Gradient Descent")

# Plot loss values
plt.subplot(1, 2, 2)
plt.plot(loss_values, "g-")
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Loss Convergence")

plt.tight_layout()

get_image(plt)
```

The stochastic gradient descent algorithm converges to the optimal parameter vector $\mathbf{\theta}$ that minimizes the loss function (MSE). The regression line fits the data points well, and the loss decreases over epochs until convergence.

</details>

Overall, we can compare the normal equation, gradient descent, and stochastic gradient descent for linear regression based on their computational complexity and convergence properties. The choice of algorithm depends on the dataset size, computational resources, and optimization requirements.

| Method                            | Runtime Complexity | Convergence                | Accuracy                                                                      |
| --------------------------------- | ------------------ | -------------------------- | ----------------------------------------------------------------------------- |
| Normal Equation                   | $O(n^3)$           | Instant (non-iterative)    | Exact solution for convex problems                                            |
| Gradient Descent                  | $O(kn^2)$          | Linear convergence rate    | Can achieve high accuracy with proper learning rate and sufficient iterations |
| Stochastic Gradient Descent (SGD) | $O(kn)$            | Sublinear convergence rate | Can approach optimal solution, but may oscillate around it                    |

Below we present an interactive example of linear regression using gradient descent. Suppose we have a dataset with one feature and one target value. The linear regression hypothesis function can be represented as:

$$
\begin{equation} \label{eq:example-hypothesis}
\hat{y} = h_{\theta}(x) = \theta_0 + \theta_1 x,
\end{equation}
$$

where $\theta_0$ is the bias term and $\theta_1$ is the weight of the feature $x$, and $\bm{\theta} = [\theta_0, \theta_1]$ is the parameter vector. Next, we use MSE $\eqref{eq:mse}$ as the loss function:

$$
L(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2,
$$

where $m$ is the number of examples, $\hat{y}^{(i)}$ is the predicted value for example $i$, and $y^{(i)}$ is the actual target value. Next, we find the gradient of the loss function with respect to the parameter vector $\bm{\theta}$.

- **Gradient with respect to $\theta_0$**:

$$
\begin{aligned}
L(\theta) &= \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 \\
L(\theta) &= \frac{1}{m} \sum_{i=1}^{m} (\theta_0 + \theta_1 x^{(i)} - y^{(i)})^2 \\
\frac{\partial L(\theta)}{\partial \theta_0} &= \frac{2}{m} \sum_{i=1}^{m} (\theta_0 + \theta_1 x^{(i)} - y^{(i)}).
\end{aligned}
$$

- **Gradient with respect to $\theta_1$**:

$$
\begin{aligned}
L(\theta) &= \frac{1}{m} \sum_{i=1}^{m} (\theta_0 + \theta_1 x^{(i)} - y^{(i)})^2 \\
\frac{\partial L(\theta)}{\partial \theta_1} &= \frac{2}{m} \sum_{i=1}^{m} (\theta_0 + \theta_1 x^{(i)} - y^{(i)}) x^{(i)}.
\end{aligned}
$$

The gradient descent update rule is then applied to update the parameter vector $\bm{\theta}$:

$$
\begin{aligned}
\theta_0^{(t+1)} &= \theta_0^{(t)} - \alpha \frac{2}{m} \sum_{i=1}^{m} (\theta_0^{(t)} + \theta_1^{(t)} x^{(i)} - y^{(i)}), \\
\theta_1^{(t+1)} &= \theta_1^{(t)} - \alpha \frac{2}{m} \sum_{i=1}^{m} (\theta_0^{(t)} + \theta_1^{(t)} x^{(i)} - y^{(i)}) x^{(i)},
\end{aligned}
$$

where $\alpha$ is the learning rate.

Since we are only dealing with two parameters $\theta_0$ and $\theta_1$, we can treat the hypothesis function $\eqref{eq:example-hypothesis}$ as a linear function in the form of $y = mx + b$ on a $xy$-plane, where $m = \theta_1$ and $b = \theta_0$.

The interactive plot below allows you to adjust the learning rate and perform each iteration of gradient descent to find the optimal parameter vector $\bm{\theta}$ that minimizes the loss function.

```component

{
    componentName: "linearRegression"
}

```

### Convergence Analysis of Gradient Descent

The convergence of gradient descent depends on the choice of the learning rate $\alpha$ and the properties of the loss function. For the following analysis, we only consider any loss function that is **convex** and **differentiable**.

<blockquote class="definition">

A function $ f: \mathbb{R}^n \to \mathbb{R} $ is **convex** if, for any two points $ x, y \in \mathbb{R}^n $ and any $ \lambda \in [0, 1] $, the following inequality holds:

$$
\begin{equation} \label{eq:convex}
 f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y)
\end{equation}
$$

In other words, the line segment connecting any two points on the graph of the function lies above or on the graph itself.

</blockquote>

<blockquote class="definition">

A function $ f $ is **differentiable** at a point $ x \in \mathbb{R}^n $ if the following limit exists:

$$
\begin{equation} \label{eq:differentiable}
\nabla f(x) = \lim_{h \to 0} \frac{f(x + h) - f(x) - \langle \nabla f(x), h \rangle}{\|h\|}
\end{equation}
$$

where $ \nabla f(x) $ denotes the gradient vector of $ f $ at $ x $, and $ \langle \nabla f(x), h \rangle $ represents the dot product of $ \nabla f(x) $ and $ h $.

</blockquote>

#### Fixed Learning Rate

coming soon...
