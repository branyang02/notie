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

### Equations inside Lists

1. Let us have an equation here:

   $$
   \begin{equation} \label{eq:cases}
   f(x) = \begin{cases}
   1 & \text{if } x > 0 \\
   0 & \text{if } x = 0 \\
   -1 & \text{if } x < 0
   \end{cases}
   \end{equation}
   $$

   Also an `align` environment:

   $$
   \begin{align}
   y &= 2 + 2 \label{eq:align-1} \\
   z &= 3 + 3 \label{eq:align-2} \\
   t &= \begin{cases}
   a &= 4 + 4 \\
   b + 10 &= 5 + 5
   \end{cases} \label{eq:align-3} \\
   f &= \begin{smallmatrix}
   a & b \\
   c & d
   \end{smallmatrix} \label{eq:align-4} \\
   g &= \begin{bmatrix}
   a+10 &= 8 + 8 \\
   b &= 9 + 9
   \end{bmatrix}. \label{eq:align-5}
   \end{align}
   $$

   Let us refer to the equation $\eqref{eq:cases}$, and the `align` environment $\eqref{eq:align-1}$, $\eqref{eq:align-2}$, $\eqref{eq:align-3}$, $\eqref{eq:align-4}$, $\eqref{eq:align-5}$.

$$
\begin{align}
\mathbf{a} \cdot \mathbf{b} &= \sum_{i=1}^{n} a_i b_i \label{eq:dot-product} \\
\mathbf{A}\mathbf{x} &= \mathbf{b} \\
\mathbf{C} &= \mathbf{A}\mathbf{B} \quad \text{where} \quad C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj} \\
\det(\mathbf{A}) &= a_{11}a_{22} - a_{12}a_{21} \label{eq:2x2-determinant} \\
\det(\mathbf{A}) &= a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31}) \label{eq:3x3-determinant} \\
\mathbf{A}^{-1} &= \frac{1}{\det(\mathbf{A})}\begin{pmatrix}
a_{22} & -a_{12} \\
-a_{21} & a_{11}
\end{pmatrix} \label{eq:2x2-inverse} \\
\mathbf{A}\mathbf{v} &= \lambda\mathbf{v} \label{eq:eigenvalue} \\
\|\mathbf{x}\|_2 &= \sqrt{\sum_{i=1}^{n} x_i^2} \\
\text{tr}(\mathbf{A}) &= \sum_{i=1}^{n} a_{ii} \\
x_i &= \frac{\det(\mathbf{A}_i)}{\det(\mathbf{A})} \label{eq:cramers-rule} \\
RREF(\mathbf{A}) &= \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix} \label{eq:rref} \\
A &= \begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{pmatrix} \label{eq:matrix} \\
\nabla \cdot \mathbf{E} &= \frac{\rho}{\varepsilon_0} \label{eq:gauss-law} \\
\nabla \times \mathbf{B} &= \mu_0 \left(\mathbf{J} + \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}\right) \label{eq:ampere-maxwell} \\
\frac{\partial}{\partial t} \left(\frac{1}{2} \int_V \rho \mathbf{v}^2 dV\right) &= -\oint_S p\mathbf{v} \cdot d\mathbf{A} + \int_V \rho \mathbf{g} \cdot \mathbf{v} dV + \int_V \mathbf{f} \cdot \mathbf{v} dV \label{eq:energy-equation} \\
\hat{H} \psi(\mathbf{r}, t) &= i\hbar \frac{\partial}{\partial t} \psi(\mathbf{r}, t) \\
R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} &= \frac{8\pi G}{c^4}T_{\mu\nu} \label{eq:einstein-field}
\end{align}
$$

Let's reference equations $\eqref{eq:dot-product}$, $\eqref{eq:2x2-determinant}$, $\eqref{eq:3x3-determinant}$, $\eqref{eq:2x2-inverse}$, $\eqref{eq:eigenvalue}$, $\eqref{eq:cramers-rule}$, and $\eqref{eq:rref}$ in the text.

Let's also reference $\eqref{eq:gauss-law}$, $\eqref{eq:ampere-maxwell}$, $\eqref{eq:energy-equation}$, and $\eqref{eq:einstein-field}$
