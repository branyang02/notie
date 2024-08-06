# Example: Auto Equation Numbering

**notie** defines a "section" as a block of text under a heading starting with exactly two hashes (`##`). This is a limitation of the current implementation and may change in the future.

## Usage

For example, `Usage` and `Section 2` are both sections. Any command that uses automatic numbering will be numbered correctly within sections. These are the commands that support automatic numbering:

- `\begin{equation} ... \end{equation}`
- `\begin{align} ... \end{align}`
- `\begin{alignat} ... \end{alignat}`
- `\begin{gather} ... \end{gather}`

For further details, checkout Katex's [documentation](https://katex.org/docs/supported.html).

Since this is a section, let's start with a simple equation using the `equation` environment:

$$
\begin{equation}
f(x) = x^2.
\end{equation}
$$

We can then reference Equation $\eqref{1.1}$ in the text. Next, let's consider a more complex scenario with multiple equations using the `align` environment:

$$
\begin{align}
f(x) &= x^2 + 2x + 1  \\
&= (x + 1)^2.
\end{align}
$$

We can individually reference Equation $\eqref{1.2}$ and Equation $\eqref{1.3}$ in the text.

## Section 2

This is a new section. Let's start with a simple equation using the `equation` environment:

$$
\begin{equation}
g(x) = x^3.
\end{equation}
$$

### Subsection 2.1

A subsection does not affect the numbering of equations. For example, let's consider a more complex scenario with multiple equations using the `align` environment:

$$
\begin{align}
g(x) &= x^3 + 3x^2 + 3x + 1  \\
&= (x + 1)^3.
\end{align}
$$

These equations are numbered $\eqref{2.2}$ and $\eqref{2.3}$.

## Gradient Descent Definition

Gradient Descent is an iterative optimization algorithm used to minimize a differentiable function. It operates by taking steps proportional to the negative of the gradient of the function at the current point.

The algorithm can be defined as follows:

1. Start with an initial point $x^{(0)}$.
2. For each iteration $k = 0, 1, 2, ...$:
   a. Compute the gradient of the function at the current point: $\nabla f(x^{(k)})$
   b. Update the position using the following rule:
   $$x^{(k+1)} = x^{(k)} - t\nabla f(x^{(k)})$$
   where $t > 0$ is the step size (also called the learning rate).
3. Repeat step 2 until a stopping criterion is met (e.g., number of iterations, small change in function value, or small gradient norm).

### Gradient Descent Convergence Analysis

Let's analyze the convergence of gradient descent for a convex function $f: \mathbb{R}^n \to \mathbb{R}$. We assume $f$ is differentiable and its gradient is Lipschitz continuous with constant $L > 0$.

We begin by considering the update rule for gradient descent:

$$
\begin{equation}
x^{(k+1)} = x^{(k)} - t\nabla f(x^{(k)})
\end{equation}
$$

where $t > 0$ is the step size and $k$ is the iteration number.

Let $x^*$ be the optimal point. We can bound the distance to the optimum after one step:

$$
\begin{align}
\|x^{(k+1)} - x^*\|_2^2 &= \|x^{(k)} - t\nabla f(x^{(k)}) - x^*\|_2^2   \\
&= \|x^{(k)} - x^*\|_2^2 - 2t\langle \nabla f(x^{(k)}), x^{(k)} - x^* \rangle + t^2\|\nabla f(x^{(k)})\|_2^2
\end{align}
$$

Now, we use the convexity of $f$ to lower bound the inner product term:

$$
\begin{equation}
f(x^*) \geq f(x^{(k)}) + \langle \nabla f(x^{(k)}), x^* - x^{(k)} \rangle
\end{equation}
$$

Rearranging $\eqref{3.4}$,

$$
\begin{equation}
\langle \nabla f(x^{(k)}), x^{(k)} - x^* \rangle \geq f(x^{(k)}) - f(x^*)
\end{equation}
$$

Substituting $\eqref{3.5}$ into $\eqref{3.3}$:

$$
\begin{equation}
\|x^{(k+1)} - x^*\|_2^2 \leq \|x^{(k)} - x^*\|_2^2 - 2t(f(x^{(k)}) - f(x^*)) + t^2\|\nabla f(x^{(k)})\|_2^2
\end{equation}
$$

Now, we use the Lipschitz continuity of $\nabla f$:

$$
\begin{align}
f(x^{(k)}) - f(x^*) &\leq \langle \nabla f(x^{(k)}), x^{(k)} - x^* \rangle - \frac{1}{2L}\|\nabla f(x^{(k)}) - \nabla f(x^*)\|_2^2   \\
&\leq \langle \nabla f(x^{(k)}), x^{(k)} - x^* \rangle - \frac{1}{2L}\|\nabla f(x^{(k)})\|_2^2
\end{align}
$$

where we used $\nabla f(x^*) = 0$ in the last step.

Rearranging $\eqref{3.8}$:

$$
\begin{equation}
\|\nabla f(x^{(k)})\|_2^2 \leq 2L(f(x^{(k)}) - f(x^*))
\end{equation}
$$

Substituting $\eqref{3.9}$ into $\eqref{3.6}$:

$$
\begin{equation}
\|x^{(k+1)} - x^*\|_2^2 \leq \|x^{(k)} - x^*\|_2^2 - 2t(1 - Lt)(f(x^{(k)}) - f(x^*))
\end{equation}
$$

If we choose $t \leq \frac{1}{L}$, then $1 - Lt \geq 0$, and we can rearrange $\eqref{3.10}$:

$$
\begin{equation}
f(x^{(k)}) - f(x^*) \leq \frac{\|x^{(k)} - x^*\|_2^2 - \|x^{(k+1)} - x^*\|_2^2}{2t(1 - Lt)}
\end{equation}
$$

Summing $\eqref{3.11}$ over $k = 0, 1, \ldots, K-1$:

$$
\begin{align}
\sum_{k=0}^{K-1} (f(x^{(k)}) - f(x^*)) &\leq \frac{\|x^{(0)} - x^*\|_2^2 - \|x^{(K)} - x^*\|_2^2}{2t(1 - Lt)}   \\
&\leq \frac{\|x^{(0)} - x^*\|_2^2}{2t(1 - Lt)}
\end{align}
$$

By convexity of $f$, we have:

$$
\begin{equation}
f\left(\frac{1}{K}\sum_{k=0}^{K-1} x^{(k)}\right) - f(x^*) \leq \frac{1}{K}\sum_{k=0}^{K-1} (f(x^{(k)}) - f(x^*))
\end{equation}
$$

Combining $\eqref{3.14}$ and $\eqref{3.13}$, we get our final convergence result:

$$
\begin{equation}
f\left(\frac{1}{K}\sum_{k=0}^{K-1} x^{(k)}\right) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|_2^2}{2t(1 - Lt)K}
\end{equation}
$$

This bound shows that the average of the iterates converges to the optimal value at a rate of $O(1/K)$, where $K$ is the number of iterations.

## Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a variant of gradient descent that uses a random sample of data to compute an estimate of the gradient. This method is particularly useful when the objective function can be decomposed into a sum of individual terms, as is common in machine learning.

The algorithm can be defined as follows:

1. Start with an initial point $x^{(0)}$.
2. For each iteration $k = 0, 1, 2, ...$:
   a. Sample a data point $i$ uniformly at random.
   b. Compute the gradient of the $i$-th term of the objective function: $\nabla f_i(x^{(k)})$
   c. Update the position using the following rule:
   $$x^{(k+1)} = x^{(k)} - t\nabla f_i(x^{(k)})$$
   where $t > 0$ is the step size.
3. Repeat step 2 until a stopping criterion is met.

SGD has several advantages over traditional gradient descent. It is computationally efficient, especially for large datasets, as it processes only one data point per iteration. Additionally, the stochastic nature of the algorithm can help escape local minima and explore the solution space more effectively.

### Stochastic Gradient Descent Convergence Analysis

Let's analyze the convergence of stochastic gradient descent for a convex function $f: \mathbb{R}^n \to \mathbb{R}$ that can be decomposed as $f(x) = \frac{1}{m}\sum_{i=1}^m f_i(x)$. We assume each $f_i$ is convex and has Lipschitz continuous gradients with constant $L > 0$.

We begin by considering the update rule for SGD:

$$
\begin{equation}
x^{(k+1)} = x^{(k)} - t_k\nabla f_{i_k}(x^{(k)})
\end{equation}
$$

where $t_k > 0$ is the step size at iteration $k$, and $i_k$ is randomly chosen from $\{1, ..., m\}$ at each iteration.

Let $x^*$ be the optimal point. We can bound the expected distance to the optimum after one step:

$$
\begin{align}
\mathbb{E}[\|x^{(k+1)} - x^*\|_2^2] &= \mathbb{E}[\|x^{(k)} - t_k\nabla f_{i_k}(x^{(k)}) - x^*\|_2^2]  \\
&= \|x^{(k)} - x^*\|_2^2 - 2t_k\mathbb{E}[\langle \nabla f_{i_k}(x^{(k)}), x^{(k)} - x^* \rangle] + t_k^2\mathbb{E}[\|\nabla f_{i_k}(x^{(k)})\|_2^2]
\end{align}
$$

Now, we use the convexity of each $f_i$ and the fact that $\mathbb{E}[\nabla f_{i_k}(x^{(k)})] = \nabla f(x^{(k)})$:

$$
\begin{equation}
\mathbb{E}[\langle \nabla f_{i_k}(x^{(k)}), x^{(k)} - x^* \rangle] = \langle \nabla f(x^{(k)}), x^{(k)} - x^* \rangle \geq f(x^{(k)}) - f(x^*)
\end{equation}
$$

For the second-order term, we can use the Lipschitz continuity of the gradients:

$$
\begin{equation}
\mathbb{E}[\|\nabla f_{i_k}(x^{(k)})\|_2^2] \leq 2L(f(x^{(k)}) - f(x^*)) + \sigma^2
\end{equation}
$$

where $\sigma^2$ is the variance of the stochastic gradients.

Combining these bounds, we get:

$$
\begin{equation}
\mathbb{E}[\|x^{(k+1)} - x^*\|_2^2] \leq \|x^{(k)} - x^*\|_2^2 - 2t_k(1 - Lt_k)(f(x^{(k)}) - f(x^*)) + t_k^2\sigma^2
\end{equation}
$$

To ensure convergence, we typically use a decreasing step size sequence satisfying $\sum_{k=1}^\infty t_k = \infty$ and $\sum_{k=1}^\infty t_k^2 < \infty$. A common choice is $t_k = \frac{c}{\sqrt{k}}$ for some constant $c > 0$.

With this step size sequence, we can show that:

$$
\begin{equation}
\mathbb{E}[f(\bar{x}^{(K)})] - f(x^*) \leq O\left(\frac{1}{\sqrt{K}}\right)
\end{equation}
$$

where $\bar{x}^{(K)} = \frac{1}{K}\sum_{k=1}^K x^{(k)}$ is the average of the iterates.

This bound shows that SGD converges to the optimal value at a rate of $O(1/\sqrt{K})$, which is slower than the $O(1/K)$ rate of full gradient descent. However, each iteration of SGD is much cheaper computationally, especially for large datasets, often making it the preferred choice in practice.
