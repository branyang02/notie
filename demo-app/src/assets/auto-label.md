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

What if we have duplicate labels within the same section?

$$
\begin{equation} \label{eq:some-equation}
E = mc^2
\end{equation}
$$

## Section 2

**notie** checks for duplicate labels within the same section. Let's copy the entire section 1 with the same labels and see what happens. But first, lets add some equations to this section before copying to see if the automatic numbering continues.

### More Equations

$$
\begin{equation}
\int_{0}^{1} x^2 dx = \frac{1}{3} \label{eq:integral}
\end{equation}
$$

Let's reference equation $\eqref{eq:integral}$ in the text.

$$
\begin{align}
\int_{0}^{1} x^2 dx &= \frac{1}{3} \label{eq:integral-align} \\
\int_{0}^{1} x^3 dx &= \frac{1}{4} \label{eq:integral-align-2}
\end{align}
$$

Reference equations $\eqref{eq:integral-align}$ and $\eqref{eq:integral-align-2}$ in the text.

### Copied from Section 1

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

What if we have duplicate labels within the same section?

$$
\begin{equation} \label{eq:some-equation}
E = mc^2
\end{equation}
$$

## More Text to extend scrolling

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc ullamcorper sem suscipit euismod condimentum. Etiam ut risus consectetur, tempus est sit amet, hendrerit arcu. Morbi molestie magna sed urna pulvinar viverra. Proin sit amet justo dolor. Cras id augue non massa mattis pretium. Phasellus cursus eu est in ullamcorper. Vivamus eu vulputate turpis. Duis sodales, ante et placerat iaculis, mauris leo viverra elit, et viverra turpis felis ut orci. Maecenas ut dui venenatis turpis dictum aliquam non a nibh. Pellentesque nulla leo, suscipit et ultricies ut, lobortis vel eros. Maecenas eget massa in dolor ornare tincidunt. Sed nec fermentum elit. Praesent ullamcorper enim non lectus lacinia, eu mollis dui maximus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae;

Praesent augue mi, venenatis eu vulputate a, blandit eu quam. Proin diam justo, porta id rutrum at, porta a nulla. Aenean dictum rhoncus mi vel imperdiet. Ut finibus mi sit amet tempor sodales. Nunc sed purus maximus, auctor odio et, porta lorem. Vivamus lacinia purus elit, eleifend aliquam lectus molestie volutpat. Nunc ac gravida dui, nec lacinia arcu. Quisque volutpat velit sed enim ornare condimentum. Donec pellentesque posuere ante vitae porttitor. Phasellus finibus rutrum elit, quis accumsan nisl laoreet in. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Sed et justo est. Nunc sit amet orci erat. In ut justo sodales, iaculis libero sit amet, hendrerit ex. Vestibulum maximus vel nulla id aliquet.

Vivamus sollicitudin vulputate cursus. Ut pretium nisi at urna tincidunt, nec lobortis elit fringilla. Aliquam pharetra dignissim laoreet. Sed euismod diam id ullamcorper pellentesque. Nullam eu lectus erat. Pellentesque vitae magna consequat, lobortis sem ut, tempor velit. Sed nec sapien eget dolor interdum ultricies ac vel arcu. Suspendisse eget auctor magna. Curabitur volutpat dolor nisi, nec mattis ex feugiat vitae. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed purus diam, pretium sit amet aliquet a, elementum at neque. Duis in neque dictum, molestie felis et, finibus metus. Proin vel turpis urna. Donec vestibulum lacus est. Suspendisse efficitur ligula purus, eget laoreet turpis finibus nec. Maecenas ex quam, semper quis risus nec, pretium vulputate ipsum.

Morbi dignissim massa ac erat sagittis accumsan. Pellentesque elementum metus et velit fermentum sodales. Sed faucibus, lacus vitae ornare convallis, purus sapien euismod metus, at tempus ante arcu quis turpis. In sodales elit fermentum sapien hendrerit rhoncus. Duis ultrices sagittis dolor eu consectetur. Proin massa nisl, sagittis vitae turpis in, fringilla mollis neque. Aenean ante elit, eleifend quis nulla ac, sagittis condimentum dui. Suspendisse ac odio at massa viverra sodales. Mauris sollicitudin elit augue, quis scelerisque massa viverra sed. Etiam arcu nulla, rhoncus quis facilisis ac, sollicitudin at odio. Morbi rhoncus tellus vulputate nulla lobortis, vel facilisis odio ultricies. Donec vitae dui nec velit maximus dignissim. Mauris quis egestas neque, eu egestas ligula. Ut efficitur maximus placerat. Donec elementum dapibus pharetra. Nunc augue tortor, dapibus ut tempor nec, imperdiet at mi.

Cras vel massa posuere, maximus lectus eu, commodo mi. Nam eleifend eleifend lacus. Proin id hendrerit nibh, ut elementum dui. Fusce finibus egestas ultricies. Cras ut velit porttitor, porta velit sed, laoreet diam. Donec vitae nibh gravida, malesuada ligula vel, tempor libero. Sed suscipit sollicitudin tempus. Nulla venenatis imperdiet pellentesque. Duis viverra arcu at eros mattis vehicula. Nam pulvinar ornare risus, sed dignissim eros viverra non. Morbi sagittis erat in quam auctor tempor. Phasellus ex purus, sagittis eget luctus id, eleifend eget sem. Quisque fermentum facilisis lacinia.
