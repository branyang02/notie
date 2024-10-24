# Title

## Section 1

$$
\begin{equation} \label{eq:split}
\begin{split}
x &= 1 + 1 \\
&= 2
\end{split}
\end{equation}
$$

1. List 1:
   Let us have an equation here:

   $$
   \begin{equation} \label{eq:1}
   \begin{split}
   x &= 1 + 1 \\
   &= 2
   \end{split}
   \end{equation}
   $$

   Also an align environment:

   $$
   \begin{align}
   y &= 2 + 2 \label{eq:3} \\
   z &= 3 + 3 \label{eq:4} \\
   t &= \begin{cases}
   a &= 4 + 4 \\
   b &= 5 + 5
   \end{cases} \label{eq:cases} \\
   f &= \begin{smallmatrix}
   a &= 6 + 6 \\
   b &= 7 + 7
   \end{smallmatrix} \label{eq:smallmatrix} \\
   g &= \begin{bmatrix}
   a &= 8 + 8 \\
   b &= 9 + 9
   \end{bmatrix}. \label{eq:bmatrix}
   \end{align}
   $$

2. List 2:
   Let us have an equation here:
   $$
   \begin{equation} \label{eq:2}
   y = 2 + 2
   \end{equation}
   $$
   Let us refer to the equation $\eqref{eq:1}$.

Let us reference both equations $\eqref{eq:1}$ and $\eqref{eq:2}$.

Let us reference the align environment $\eqref{eq:3}$ and $\eqref{eq:4}$, $\eqref{eq:split}$

Let us reference the cases environment $\eqref{eq:cases}$
Let us reference the smallmatrix environment $\eqref{eq:smallmatrix}$
Let us reference the bmatrix environment $\eqref{eq:bmatrix}$
