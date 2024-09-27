# **Ordinary Differential Equations**

<span class="subtitle">
Fall 2024 | Author: Brandon Yang
</span>

---

## Introduction to ODEs

An **ordinary differential equation (ODE)** is an equation that contains one or more functions of one independent variable and its derivatives. For example, the following are ODEs:

<blockquote class="note">

An ODE is an equation that contains one or more functions of one independent variable and its derivatives.

</blockquote>

```desmos

y = mx + b
y = 2x^2 + 3x + 1
y = -x^3 + 2x^2 - 3x + 4

```

```desmos

y = \sin(x^2) + \frac{x^2}{10}

```

$$
\begin{aligned}
&y = y' \\
&y'' + \sin(y) = 0 \\
&e^x y'  + \cos(x) = 0,
\end{aligned}
$$

where $y$ is the dependent variable and $x$ is the independent variable. The general form of an ODE is:

$$
F(x, y, y', y'', \ldots, y^{(n)}) = 0,
$$

where $F$ is a function of $x$ and $y$ and its derivatives.

<blockquote class="important">

We use the notation $y^{(n)}$ to denote the $n^{th}$ **derivative** of $y$, and we use the standard power notation $y^n$ to denote the $n^{th}$ power of $y$.

</blockquote>

<details open><summary>Solving an ODE</summary>

Solve for $y$:

$$
y'' = 1.
$$

We solve by integrating both sides:

$$
\begin{aligned}
y' &= \int 1 \, dx = x + C_1, \\
y &= \int x + C_1 \, dx = \frac{1}{2} x^2 + C_1 x + C_2.
\end{aligned}
$$

Given that our solution includes two constants, we have infinitely many solutions.

</details>

### Classification of ODEs

<span class="subtitle">

Section 1.3 in BOYCE, DIPRIMA.

</span>

There are two ways to classify ODEs:

1. **By order**
2. **By linearity**

#### By Order

We can classify ODEs **by order**. For example:

$$
\begin{aligned}
y = y' &\quad \text{(1st order)} \\
y'' = 1 &\quad \text{(2nd order)} \\
y'' + y' = 5 &\quad \text{(2nd order)} \\
yy' = 1 &\quad \text{(1st order)} \\
y''' \ln(y)  + y^2 = \sin(y') &\quad \text{(3rd order)},
\end{aligned}
$$

where the order of an ODE is the highest derivative present.

<blockquote class="definition">

An $n^{th}$ order ODE is of the form:

$$
\begin{equation} \label{eq:ode}
F\left(x, y, y', y'', \ldots, y^{(n)} \right) = 0.
\end{equation}
$$

</blockquote>

#### By Linearity

We can also classify ODEs **by linearity**. For example:

$$
F(y, y') = y' - y
$$

is a linear ODE. We treat each $y^{(n)}$ term as a variable: $y'$ and $y$, and therefore, it can be written as a linear combination of these variables.

For a non-linear ODE, suppose we have:

$$
F(y, y', y'') = y'' + 5y^2 - \sin(x) = 0.
$$

In this case, we have three separate variables: $y$, $y'$, and $y''$. However, the term $5y^2$ has a power of 2, which makes the entire equation non-linear. We call this equation **non-linear in $y$**.

Another example is:

$$
5\ln(y') + y - \tan(x) = 0.
$$

This function is linear in $y$, but non-linear in $y'$, since the term $\ln(y')$ is non-linear.

Another example is:

$$
yy' + 10x = 0.
$$

This function is linear, since we treat each variable $y$ and $y'$ as separate variables one at the time, and both $y$ and $y'$ are linear.

<blockquote class="definition">

A **linear ODE** is of the form:

$$
\begin{equation} \label{eq:linear_ode}
a_n(x) y^{(n)} + a_{n-1}(x) y^{(n-1)} + \ldots + a_1(x) y' + a_0(x) y = F(x),
\end{equation}
$$

where $a_0(x), \ldots, a_n(x)$, and $F(x)$ are arbitrary differentiable functions that do not need to be linear.

</blockquote>

<details open><summary>Verify ODEs</summary>

Suppose we have the following problem:

Verify $y = x$ satisfies the ODE $y'=1$.

<blockquote class="proof">

We solve this by taking the derivative of $y$:

$$
y' = 1.
$$

Since $y' = 1$, we have verified that $y = x$ satisfies the ODE.

</blockquote>

Verify that the given function $y$ is a solution of the ODE:

$$
y' - 2ty = 1, \quad y = e^{t^2}\int_0^t e^{-s^2} \, ds + e^{t^2}.
$$

<blockquote class="proof">

We take the derivative of $y$:

$$
y' = \frac{d}{dt} \left( e^{t^2}\int_0^t e^{-s^2} \, ds + e^{t^2} \right)
$$

We use the product rule in $\eqref{eq:productrule}$:

$$
e^{t^2} \cdot \frac{d}{dt} \int_0^t e^{-s^2} \, ds + \frac{d}{dt} e^{t^2} \cdot \int_0^t e^{-s^2} \, ds + \frac{d}{dt} e^{t^2}.
$$

Solving the first term using the Fundamental Theorem of Calculus in $\eqref{eq:ftc2}$:

$$
\begin{aligned}
e^{t^2} \cdot \frac{d}{dt} \int_0^t e^{-s^2} \, ds &= e^{t^2} \cdot e^{-t^2} \cdot 1 = 1, \\
\end{aligned}
$$

Similarly, solving the second term using the chain rule in $\eqref{eq:chainrule}$:

$$
\frac{d}{dt} e^{t^2} \cdot \int_0^t e^{-s^2} \, ds = 2te^{t^2} \cdot \int_0^t e^{-s^2} \, ds.
$$

Finally, solving the third term:

$$
\frac{d}{dt} e^{t^2} = 2te^{t^2}.
$$

Combining all terms:

$$
\begin{aligned}
y' &= 1 + 2te^{t^2} \cdot \int_0^t e^{-s^2} \, ds + 2te^{t^2}
\end{aligned}
$$

Finally, we substitute $y'$ back into the ODE:

$$
\begin{aligned}
y' - 2ty &= 1 \\
1 + \left(2te^{t^2} \cdot \int_0^t e^{-s^2} \, ds + 2te^{t^2}\right) - 2t\left( e^{t^2}\int_0^t e^{-s^2} \, ds + e^{t^2}\right) &= 1 \\
1 + \left(2te^{t^2} \cdot \int_0^t e^{-s^2} \, ds + 2te^{t^2}\right) - \left(2t e^{t^2}\int_0^t e^{-s^2} \, ds - 2te^{t^2}\right) &= 1 \\
1 &= 1.
\end{aligned}
$$

Therefore, the function $y$ satisfies the ODE.

</blockquote>

</details>

## Background

### Review for Calculus

1. Product Rule for derivatives:

$$
\begin{equation} \label{eq:productrule}
(fg)' = f'g + fg'.
\end{equation}
$$

2. Chain Rule for derivatives:

$$
\begin{equation} \label{eq:chainrule}
\frac{d}{dx} \left[ f(g(x)) \right] = f'(g(x)) g'(x).
\end{equation}
$$

3. Fundamental Theorem of Calculus:

$$
\begin{equation} \label{eq:ftc}
\frac{d}{dx} \int f(x) \, dx = f(x).
\end{equation}
$$

$$
\begin{equation} \label{eq:ftc2}
\frac{d}{dx} \int_a^{g(x)} f(x) \, dx = f(g(x)) g'(x).
\end{equation}
$$

$$
\begin{equation} \label{eq:ftc3}
\frac{d}{dx} \int_{h(x)}^{g(x)} f(x) \, dx = f(g(x)) g'(x) - f(h(x)) h'(x).
\end{equation}
$$

<details open><summary>Examples</summary>

<blockquote class="problem">

For example, suppose we want to evaluate the following derivative:

$$
\frac{d}{dx} \cos(x^2 + 1).
$$

</blockquote>

<blockquote class="proof">

We use the chain rule from $\eqref{eq:chainrule}$:

$$
\begin{aligned}
\frac{d}{dx} \cos(x^2 + 1) &= -\sin(x^2 + 1) \cdot 2x \\
&= -2x \sin(x^2 + 1).
\end{aligned}
$$

</blockquote>

<blockquote class="problem">

Suppose we want to evaluate the following derivative:

$$
\frac{d}{dx} \left(x \cdot \cos(x^2 + 1)\right).
$$

</blockquote>

<blockquote class="proof">

We use the product rule from $\eqref{eq:productrule}$:

$$
\begin{aligned}
\frac{d}{dx} \left(x \cdot \cos(x^2 + 1)\right) &= 1 \cdot \cos(x^2 + 1) + x \cdot -\sin(x^2 + 1) \cdot 2x \\
&= \cos(x^2 + 1) - 2x^2 \sin(x^2 + 1).
\end{aligned}
$$

</blockquote>

<blockquote class="problem">

Suppose we want to evaluate the following integral:

$$
\frac{d}{dx} \int_1^2 \left( \tan(x+100) - x \right) \, dx.
$$

</blockquote>

<blockquote class="proof">

Since we are solving for a definite integral, we know from $\eqref{eq:ftc3}$ that the solution is a constant:

$$
\int_1^2 \left( \tan(x+100) - x \right) \, dx = C,
$$

where $C$ is a constant. Therefore, the derivative of a constant is zero:

$$
\frac{d}{dx} \int_1^2 \left( \tan(x+100) - x \right) \, dx = 0.
$$

</blockquote>

<blockquote class="problem">

Suppose we have the following derivative:

$$
\frac{d}{dx} \int_0^x \sin^{-1}(s^2+1) \, ds.
$$

</blockquote>

<blockquote class="proof">

We use the Fundamental Theorem of Calculus from $\eqref{eq:ftc2}$:

$$
\begin{aligned}
\frac{d}{dx} \int_0^x \sin^{-1}(s^2+1) \, ds &= \sin^{-1}(x^2+1) \cdot 1 \\
&= \sin^{-1}(x^2+1).
\end{aligned}
$$

</blockquote>

<blockquote class="problem">

Suppose we have the following derivative:

$$
\frac{d}{dx} \int_{x^2 - 1}^{10} \cos(s) \, ds.
$$

</blockquote>

<blockquote class="proof">

We use the Fundamental Theorem of Calculus from $\eqref{eq:ftc3}$:

$$
\begin{aligned}
\frac{d}{dx} \int_{x^2 - 1}^{10} \cos(s) \, ds &= \cos(10) \cdot 0 - \cos(x^2 - 1) \cdot 2x \\
&= -2x \cos(x^2 - 1).
\end{aligned}
$$

</details>
