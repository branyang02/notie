# Example: Automatic Equation Numbering

## Section 1

Let use the `\begin{equation}` environment and label it.

$$
\begin{equation} \label{eq:quadratic}
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\end{equation}
$$

Reference equation $\eqref{eq:quadratic}$ in the text, or $\ref{eq:quadratic}$.

Proof of quadratic formula, and label each step in the `\begin{align}` environment.

$$
\begin{align}
ax^2 + bx + c &= 0  \label{eq:quadratic-general} \\
x^2 + \frac{b}{a}x + \frac{c}{a} &= 0  \label{eq:quadratic-normalized} \\
x^2 + \frac{b}{a}x &= -\frac{c}{a}  \label{eq:quadratic-half} \\
x^2 + \frac{b}{a}x + \left(\frac{b}{2a}\right)^2 &= -\frac{c}{a} + \left(\frac{b}{2a}\right)^2  \label{eq:quadratic-complete} \\
\left(x + \frac{b}{2a}\right)^2 &= \frac{b^2 - 4ac}{4a^2}  \label{eq:quadratic-squared} \\
x + \frac{b}{2a} &= \pm \sqrt{\frac{b^2 - 4ac}{4a^2}}  \label{eq:quadratic-sqrt} \\
x &= -\frac{b}{2a} \pm \sqrt{\frac{b^2 - 4ac}{4a^2}}  \label{eq:quadratic-solve} \\
x &= \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \label{eq:quadratic-final}
\end{align}
$$

Reference equations $\eqref{eq:quadratic-general}$, $\eqref{eq:quadratic-normalized}$, $\eqref{eq:quadratic-half}$, $\eqref{eq:quadratic-complete}$, $\eqref{eq:quadratic-squared}$, $\eqref{eq:quadratic-sqrt}$, $\eqref{eq:quadratic-solve}$, and $\eqref{eq:quadratic-final}$ in the text.

Let's go back to using no automatic numbering with `begin{align*}`.

$$
\begin{align*}
ax^2 + bx + c &= 0 \\
x^2 + \frac{b}{a}x + \frac{c}{a} &= 0 \\
x^2 + \frac{b}{a}x &= -\frac{c}{a} \\
x^2 + \frac{b}{a}x + \left(\frac{b}{2a}\right)^2 &= -\frac{c}{a} + \left(\frac{b}{2a}\right)^2 \\
\left(x + \frac{b}{2a}\right)^2 &= \frac{b^2 - 4ac}{4a^2} \\
x + \frac{b}{2a} &= \pm \sqrt{\frac{b^2 - 4ac}{4a^2}} \\
x &= -\frac{b}{2a} \pm \sqrt{\frac{b^2 - 4ac}{4a^2}} \\
x &= \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\end{align*}
$$

And then use the `align` environment again.

$$
\begin{align}
A &= B + C \label{eq:abc} \\
D &= E + F \label{eq:def}
\end{align}
$$

Reference equations $\eqref{eq:abc}$ and $\eqref{eq:def}$ in the text.

What if I use `\begin{equation*}` environment with no automatic numbering?

$$
\begin{equation*}
b = \frac{1}{2} \pm \sqrt{\frac{1}{4} - 1}
\end{equation*}
$$

And then use `\begin{equation}` environment with automatic numbering.

$$
\begin{equation} \label{eq:some-equation}
b = \frac{1}{2} \pm \sqrt{\frac{1}{4} - 1}
\end{equation}
$$

Reference equation $\eqref{eq:some-equation}$ in the text.

## Section 2

What if we have duplicate labels?

$$
\begin{equation} \label{eq:some-equation}
E = mc^2
\end{equation}
$$

We can reference all equations in the document:

- $\eqref{eq:quadratic}$
- $\eqref{eq:quadratic-general}$
- $\eqref{eq:quadratic-normalized}$
- $\eqref{eq:quadratic-half}$
- $\eqref{eq:quadratic-complete}$
- $\eqref{eq:quadratic-squared}$
- $\eqref{eq:quadratic-sqrt}$
- $\eqref{eq:quadratic-solve}$
- $\eqref{eq:quadratic-final}$
- $\eqref{eq:abc}$
- $\eqref{eq:def}$
- $\eqref{eq:some-equation}$

We don't have to label equations in the `align` environment.

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
\end{pmatrix} \label{eq:matrix}
\end{align}
$$

Let's see if other **notie** functions still work.

Let's reference equations $\eqref{eq:dot-product}$, $\eqref{eq:2x2-determinant}$, $\eqref{eq:3x3-determinant}$, $\eqref{eq:2x2-inverse}$, $\eqref{eq:eigenvalue}$, $\eqref{eq:cramers-rule}$, and $\eqref{eq:rref}$ in the text.

```python
for i in range(10):
    print("Hello World")
```

```execute-python
for i in range(10):
    print("Hello World")
```

## More Text to extend scrolling

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc ullamcorper sem suscipit euismod condimentum. Etiam ut risus consectetur, tempus est sit amet, hendrerit arcu. Morbi molestie magna sed urna pulvinar viverra. Proin sit amet justo dolor. Cras id augue non massa mattis pretium. Phasellus cursus eu est in ullamcorper. Vivamus eu vulputate turpis. Duis sodales, ante et placerat iaculis, mauris leo viverra elit, et viverra turpis felis ut orci. Maecenas ut dui venenatis turpis dictum aliquam non a nibh. Pellentesque nulla leo, suscipit et ultricies ut, lobortis vel eros. Maecenas eget massa in dolor ornare tincidunt. Sed nec fermentum elit. Praesent ullamcorper enim non lectus lacinia, eu mollis dui maximus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae;

Praesent augue mi, venenatis eu vulputate a, blandit eu quam. Proin diam justo, porta id rutrum at, porta a nulla. Aenean dictum rhoncus mi vel imperdiet. Ut finibus mi sit amet tempor sodales. Nunc sed purus maximus, auctor odio et, porta lorem. Vivamus lacinia purus elit, eleifend aliquam lectus molestie volutpat. Nunc ac gravida dui, nec lacinia arcu. Quisque volutpat velit sed enim ornare condimentum. Donec pellentesque posuere ante vitae porttitor. Phasellus finibus rutrum elit, quis accumsan nisl laoreet in. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Sed et justo est. Nunc sit amet orci erat. In ut justo sodales, iaculis libero sit amet, hendrerit ex. Vestibulum maximus vel nulla id aliquet.

Vivamus sollicitudin vulputate cursus. Ut pretium nisi at urna tincidunt, nec lobortis elit fringilla. Aliquam pharetra dignissim laoreet. Sed euismod diam id ullamcorper pellentesque. Nullam eu lectus erat. Pellentesque vitae magna consequat, lobortis sem ut, tempor velit. Sed nec sapien eget dolor interdum ultricies ac vel arcu. Suspendisse eget auctor magna. Curabitur volutpat dolor nisi, nec mattis ex feugiat vitae. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed purus diam, pretium sit amet aliquet a, elementum at neque. Duis in neque dictum, molestie felis et, finibus metus. Proin vel turpis urna. Donec vestibulum lacus est. Suspendisse efficitur ligula purus, eget laoreet turpis finibus nec. Maecenas ex quam, semper quis risus nec, pretium vulputate ipsum.

Morbi dignissim massa ac erat sagittis accumsan. Pellentesque elementum metus et velit fermentum sodales. Sed faucibus, lacus vitae ornare convallis, purus sapien euismod metus, at tempus ante arcu quis turpis. In sodales elit fermentum sapien hendrerit rhoncus. Duis ultrices sagittis dolor eu consectetur. Proin massa nisl, sagittis vitae turpis in, fringilla mollis neque. Aenean ante elit, eleifend quis nulla ac, sagittis condimentum dui. Suspendisse ac odio at massa viverra sodales. Mauris sollicitudin elit augue, quis scelerisque massa viverra sed. Etiam arcu nulla, rhoncus quis facilisis ac, sollicitudin at odio. Morbi rhoncus tellus vulputate nulla lobortis, vel facilisis odio ultricies. Donec vitae dui nec velit maximus dignissim. Mauris quis egestas neque, eu egestas ligula. Ut efficitur maximus placerat. Donec elementum dapibus pharetra. Nunc augue tortor, dapibus ut tempor nec, imperdiet at mi.

Cras vel massa posuere, maximus lectus eu, commodo mi. Nam eleifend eleifend lacus. Proin id hendrerit nibh, ut elementum dui. Fusce finibus egestas ultricies. Cras ut velit porttitor, porta velit sed, laoreet diam. Donec vitae nibh gravida, malesuada ligula vel, tempor libero. Sed suscipit sollicitudin tempus. Nulla venenatis imperdiet pellentesque. Duis viverra arcu at eros mattis vehicula. Nam pulvinar ornare risus, sed dignissim eros viverra non. Morbi sagittis erat in quam auctor tempor. Phasellus ex purus, sagittis eget luctus id, eleifend eget sem. Quisque fermentum facilisis lacinia.

## Section 4

We can still reference equations in the document:

- $\eqref{eq:quadratic}$
- $\eqref{eq:quadratic-general}$
- $\eqref{eq:matrix}$
