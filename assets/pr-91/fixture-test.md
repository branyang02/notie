# Issue 85 Review Fixture

$$
\begin{equation} \label{eq:title-sec}
t = 0
\end{equation}
$$

## Gather and Alignat Mapping

A gather environment (two rows, label on the second row):

$$
\begin{gather}
a = 1 \\
b = 2 \label{eq:gather-two}
\end{gather}
$$

An equation after the gather (must stay in sync with KaTeX numbering):

$$
\begin{equation} \label{eq:after-gather}
c = 3
\end{equation}
$$

An alignat environment with argument:

$$
\begin{alignat}{2}
d &= 4 &\quad e &= 5 \label{eq:alignat-one} \\
f &= 6 &\quad g &= 7 \label{eq:alignat-two}
\end{alignat}
$$

References: gather row two is $\eqref{eq:gather-two}$, the equation after
the gather is $\eqref{eq:after-gather}$, alignat row one is
$\eqref{eq:alignat-one}$ and row two is $\eqref{eq:alignat-two}$.

## Second Section

$$
\begin{equation} \label{eq:s2}
h = 8
\end{equation}
$$

Reference into section one from here: $\eqref{eq:after-gather}$, and local
$\eqref{eq:s2}$.
