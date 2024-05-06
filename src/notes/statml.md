# Statistical Machine Learning

## Jansen's Inequality

Jensen's inequality is a fundamental result in convex analysis that provides a lower bound on the expected value of a convex function of a random variable. The inequality states that if $f$ is a convex function and $X$ is a random variable, then the expected value of $f(X)$ is greater than or equal to the convex function of the expected value of $X$.

```tikz
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}

\begin{document}
  \begin{tikzpicture}[domain=0:4, scale=1.2]
    \draw[very thin,color=gray] (-0.1,-1.1) grid (3.9,3.9);
    \draw[->] (-0.2,0) -- (4.2,0) node[right] {$x$};
    \draw[->] (0,-1.2) -- (0,4.2) node[above] {$f(x)$};
    \draw[color=red]    plot (\x,\x)             node[right] {$f(x) =x$};
    \draw[color=blue]   plot (\x,{sin(\x r)})    node[right] {$f(x) = \sin x$};
    \draw[color=orange] plot (\x,{0.05*exp(\x)}) node[right] {$f(x) = \frac{1}{20} \mathrm e^x$};
  \end{tikzpicture}

  \begin{tikzpicture}
\begin{axis}[colormap/viridis]
\addplot3[
	surf,
	samples=18,
	domain=-3:3
]
{exp(-x^2-y^2)*x};
\end{axis}
\end{tikzpicture}

\end{document}
```

<span class="caption">
You can draw awesome plots with TikZ and PGFPlots.
</span>

<blockquote class="theorem">

If $f$ is a convex function and $X$ is a random variable, then

$$
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)],
$$

where:

- $\mathbb{E}[X]$ is the expected value of $X$.
- $\mathbb{E}[f(X)]$ is the expected value of $f(X)$.

</blockquote>

<blockquote class="proof">

Let $f$ be a convex function and $X$ be a random variable. By the definition of convexity, we have:

$$
f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y),
$$

for all $x, y \in \mathbb{R}$ and $\lambda \in [0, 1]$. Taking the expected value of both sides, we get:

$$
\begin{aligned}
\mathbb{E}[f(\lambda X + (1 - \lambda) Y)] & \leq \mathbb{E}[\lambda f(X) + (1 - \lambda) f(Y)] \\
f(\mathbb{E}[\lambda X + (1 - \lambda) Y]) & \leq \lambda f(\mathbb{E}[X]) + (1 - \lambda) f(\mathbb{E}[Y]),
\end{aligned}
$$

where $Y$ is another random variable. Since $\mathbb{E}[\lambda X + (1 - \lambda) Y] = \lambda \mathbb{E}[X] + (1 - \lambda) \mathbb{E}[Y]$, we have:

$$
f(\mathbb{E}[X]) \leq \lambda f(\mathbb{E}[X]) + (1 - \lambda) f(\mathbb{E}[Y]).
$$

Taking the expected value of both sides, we get:

$$
f(\mathbb{E}[X]) \leq \mathbb{E}[\lambda f(X) + (1 - \lambda) f(Y)].
$$

Since this holds for all $\lambda \in [0, 1]$, we have:

$$
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)].
$$

</blockquote>

<blockquote class="example">

Let $X$ be a random variable with mean $\mu$ and variance $\sigma^2$. Let $f(x) = x^2$. By Jensen's inequality, we have:

$$
\begin{aligned}
f(\mu) & = \mu^2 \\
& \leq \mathbb{E}[X^2] \\
& = \mu^2 + \sigma^2.
\end{aligned}
$$

Therefore, $\sigma^2 \geq 0$.

</blockquote>

<details open>
<summary>Coding Example</summary>

```execute-python
import numpy as np

# Generate random variable
np.random.seed(0)
X = np.random.normal(0, 1, 1000)

# Compute mean and variance
mu = np.mean(X)
sigma2 = np.var(X)

# Compute Jensen's inequality
f_mu = mu**2
f_X = np.mean(X**2)

# Print results
print(f"Mean: {mu:.2f}")
print(f"Variance: {sigma2:.2f}")
print(f"Jensen's Inequality: {f_mu:.2f} <= {f_X:.2f}")
```

</details>

As we can see from the example, Jensen's inequality provides a lower bound on the expected value of a convex function of a random variable. In this case, the variance of the random variable is non-negative, as expected.

## Jensen's Inequality in Machine Learning

Jensen's inequality is a powerful tool in machine learning for deriving bounds on the expected value of a convex function of a random variable. In particular, Jensen's inequality is often used to derive the loss functions for various machine learning algorithms.

<blockquote class="example">

Consider the logistic loss function:

$$
\ell(y, \hat{y}) = \log(1 + \exp(-y \hat{y})),
$$

where $y \in \{-1, 1\}$ is the true label and $\hat{y} \in \mathbb{R}$ is the predicted score. The logistic loss function is a convex function of the predicted score. By Jensen's inequality, we have:

$$
\begin{aligned}
\ell(y, \hat{y}) & = \log(1 + \exp(-y \hat{y})) \\
& \leq \log(1 + \exp(-y \mathbb{E}[\hat{y}])),
\end{aligned}
$$

where $\mathbb{E}[\hat{y}]$ is the expected value of the predicted score.

</blockquote>

The logistic loss function is commonly used in binary classification problems to measure the error between the true label and the predicted score. By applying Jensen's inequality, we can derive a lower bound on the expected value of the logistic loss function, which can be used to optimize the model parameters.

## Conclusion

Jensen's inequality is a fundamental result in convex analysis that provides a lower bound on the expected value of a convex function of a random variable. In machine learning, Jensen's inequality is a powerful tool for deriving bounds on the expected value of loss functions, which can be used to optimize machine learning algorithms.
