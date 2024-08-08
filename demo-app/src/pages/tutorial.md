# **notie** Turotial

## Markdown

**notie** supports most native markdown syntax. This tutorial provides an overview of the markdown syntax supported by **notie**.

### Title and headings

**notie** automatically generates the title and the table of contents based on the headings in your markdown file. **notie** support the following heading levels:

- `# Title`
- `## Heading 1`
- `### Heading 2`
- `#### Heading 3`
- `##### Heading 4`
- `###### Heading 5`

### Paragraphs

To create paragraphs, use a blank line to separate one or more lines of text.

```markdown
This is a paragraph.

This is another paragraph.
```

This is a paragraph.

This is another paragraph.

### Emphasis

You can make text **bold** or _italic_ using the following syntax:

```markdown
**bold words**, _italic words_, **_bold and italic words_**, `code`
```

**bold words**, _italic words_, **_bold and italic words_**, `code`

### Lists

You can create ordered and unordered lists using the following syntax:

```markdown
- Unordered list item 1
  - Unordered list item 1.1
- Unordered list item 2
  - Unordered list item 2.1
  - Unordered list item 2.2
- Unordered list item 3
  - Unordered list item 3.1
    - Unordered list item 3.1.1
      - Unordered list item 3.1.1.1
```

- Unordered list item 1
  - Unordered list item 1.1
- Unordered list item 2
  - Unordered list item 2.1
  - Unordered list item 2.2
- Unordered list item 3
  - Unordered list item 3.1
    - Unordered list item 3.1.1
      - Unordered list item 3.1.1.1

```markdown
1. Ordered list item 1
2. Ordered list item 2
3. Ordered list item 3
```

1. Ordered list item 1
2. Ordered list item 2
3. Ordered list item 3

## Code Blocks

**notie** supports both static code blocks, and **live**, **editable**, and **runnable** code blocks.

### Static code blocks

To create a static code block, use triple backticks followed by the language name:

```python
def hello_world():
    print("Hello, World!")

hello_world()
```

```javascript
function helloWorld() {
  console.log("Hello, World!");
}

helloWorld();
```

### Live code blocks

To create a live code block, use `execute-[language]` after the triple backticks:

```execute-python
def hello_world():
    print("Hello, World!")

hello_world()
```

```execute-javascript
function helloWorld() {
  console.log("Hello, World!");
}

helloWorld();
```

For more information on supported languages, see the [Programming](https://notie-nine.vercel.app/examples/programming) page.

#### Special Note for Python Support

**notie** additionally supports [PyTorch](https://pytorch.org/) and [Matplotlib](https://matplotlib.org/) for Python code blocks.

##### PyTorch

```execute-python
import torch

print(torch.__version__)
print(torch.cuda.is_available())

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = NeuralNetwork()
print(model)
```

##### Matplotlib

Users can use the pre-definied `get_image` function to display Matplotlib plots.

```execute-python
import matplotlib.pyplot as plt

# random data
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 35]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple plot')
plt.grid(True)
plt.tight_layout()

get_image(plt)
```

## Math

**notie** uses [KaTeX](https://katex.org/) to render math equations. You can use the following syntax to write math equations:

### Inline Math

To write inline math equations, use the following syntax:

```markdown
The Pythagorean theorem is $a^2 + b^2 = c^2$.
```

The Pythagorean theorem is $a^2 + b^2 = c^2$.

### Block Math

To write block math equations, use the following syntax:

```markdown
Let $a$, $b$, and $c$ be the coefficients of a quadratic equation $ax^2 + bx + c = 0$. The solutions to this equation are given by the quadratic formula:

$$
\begin{equation*}
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\end{equation*}
$$
```

Let $a$, $b$, and $c$ be the coefficients of a quadratic equation $ax^2 + bx + c = 0$. The solutions to this equation are given by the quadratic formula:

$$
\begin{equation*}
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\end{equation*}
$$

```markdown
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
```

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

### Equation Numbering and Referencing

**notie** generates automatic equation numbering and referencing for the `\begin{equation}` and `\begin{align}` environments based on the current **section**.

<blockquote class="definition">

A **section** in **notie** is anything under a heading level 2 (`##`). Any subsections under a heading level 3 (`###`) are considered part of the same section.

</blockquote>

For example, we are currently in the **Math** section, which is the third section in this markdown file. The first section is **Markdown**, and the second section is **Code Blocks**. Let's generate some equations and reference them.

```markdown
$$
\begin{equation}
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \label{eq:quadratic}
\end{equation}
$$

Reference equation $\eqref{eq:quadratic}$ in the text.
```

$$
\begin{equation}
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \label{eq:quadratic}
\end{equation}
$$

Reference equation $\eqref{eq:quadratic}$ in the text.

```markdown
$$
\begin{align}
ax^2 + bx + c &= 0 \label{eq:quadratic-general} \\
x^2 + \frac{b}{a}x + \frac{c}{a} &= 0 \label{eq:quadratic-normalized} \\
x^2 + \frac{b}{a}x &= -\frac{c}{a} \label{eq:quadratic-half} \\
x^2 + \frac{b}{a}x + \left(\frac{b}{2a}\right)^2 &= -\frac{c}{a} + \left(\frac{b}{2a}\right)^2 \label{eq:quadratic-complete} \\
\left(x + \frac{b}{2a}\right)^2 &= \frac{b^2 - 4ac}{4a^2} \label{eq:quadratic-squared} \\
x + \frac{b}{2a} &= \pm \sqrt{\frac{b^2 - 4ac}{4a^2}} \label{eq:quadratic-sqrt} \\
x &= -\frac{b}{2a} \pm \sqrt{\frac{b^2 - 4ac}{4a^2}} \label{eq:quadratic-solve} \\
x &= \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \label{eq:quadratic-final}
\end{align}
$$

Reference equations $\eqref{eq:quadratic-general}$, $\eqref{eq:quadratic-normalized}$, $\eqref{eq:quadratic-half}$, $\eqref{eq:quadratic-complete}$, $\eqref{eq:quadratic-squared}$, $\eqref{eq:quadratic-sqrt}$, $\eqref{eq:quadratic-solve}$, and $\eqref{eq:quadratic-final}$ in the text.
```

$$
\begin{align}
ax^2 + bx + c &= 0 \label{eq:quadratic-general} \\
x^2 + \frac{b}{a}x + \frac{c}{a} &= 0 \label{eq:quadratic-normalized} \\
x^2 + \frac{b}{a}x &= -\frac{c}{a} \label{eq:quadratic-half} \\
x^2 + \frac{b}{a}x + \left(\frac{b}{2a}\right)^2 &= -\frac{c}{a} + \left(\frac{b}{2a}\right)^2 \label{eq:quadratic-complete} \\
\left(x + \frac{b}{2a}\right)^2 &= \frac{b^2 - 4ac}{4a^2} \label{eq:quadratic-squared} \\
x + \frac{b}{2a} &= \pm \sqrt{\frac{b^2 - 4ac}{4a^2}} \label{eq:quadratic-sqrt} \\
x &= -\frac{b}{2a} \pm \sqrt{\frac{b^2 - 4ac}{4a^2}} \label{eq:quadratic-solve} \\
x &= \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \label{eq:quadratic-final}
\end{align}
$$

Reference equations $\eqref{eq:quadratic-general}$, $\eqref{eq:quadratic-normalized}$, $\eqref{eq:quadratic-half}$, $\eqref{eq:quadratic-complete}$, $\eqref{eq:quadratic-squared}$, $\eqref{eq:quadratic-sqrt}$, $\eqref{eq:quadratic-solve}$, and $\eqref{eq:quadratic-final}$ in the text.

## TikZ diagrams

You can create TikZ diagrams in your markdown file by including `tikz` after the triple backticks. For example,

```tikz
\begin{document}
\begin{tikzpicture}
    % Draw a rectangle
    \draw (0,0) rectangle (4,2);

    % Label the corners of the rectangle
    \node at (0,0) [below left] {A};
    \node at (4,0) [below right] {B};
    \node at (4,2) [above right] {C};
    \node at (0,2) [above left] {D};
\end{tikzpicture}
\end{document}
```

```tikz
\begin{document}

\begin{tikzpicture}[level/.style={sibling distance=60mm/#1}]
\node [circle,draw] (z){$n$}
  child {node [circle,draw] (a) {$\frac{n}{2}$}
    child {node [circle,draw] (b) {$\frac{n}{2^2}$}
      child {node {$\vdots$}
        child {node [circle,draw] (d) {$\frac{n}{2^k}$}}
        child {node [circle,draw] (e) {$\frac{n}{2^k}$}}
      }
      child {node {$\vdots$}}
    }
    child {node [circle,draw] (g) {$\frac{n}{2^2}$}
      child {node {$\vdots$}}
      child {node {$\vdots$}}
    }
  }
  child {node [circle,draw] (j) {$\frac{n}{2}$}
    child {node [circle,draw] (k) {$\frac{n}{2^2}$}
      child {node {$\vdots$}}
      child {node {$\vdots$}}
    }
  child {node [circle,draw] (l) {$\frac{n}{2^2}$}
    child {node {$\vdots$}}
    child {node (c){$\vdots$}
      child {node [circle,draw] (o) {$\frac{n}{2^k}$}}
      child {node [circle,draw] (p) {$\frac{n}{2^k}$}
        child [grow=right] {node (q) {$=$} edge from parent[draw=none]
          child [grow=right] {node (q) {$O_{k = \lg n}(n)$} edge from parent[draw=none]
            child [grow=up] {node (r) {$\vdots$} edge from parent[draw=none]
              child [grow=up] {node (s) {$O_2(n)$} edge from parent[draw=none]
                child [grow=up] {node (t) {$O_1(n)$} edge from parent[draw=none]
                  child [grow=up] {node (u) {$O_0(n)$} edge from parent[draw=none]}
                }
              }
            }
            child [grow=down] {node (v) {$O(n \cdot \lg n)$}edge from parent[draw=none]}
          }
        }
      }
    }
  }
};
\path (a) -- (j) node [midway] {+};
\path (b) -- (g) node [midway] {+};
\path (k) -- (l) node [midway] {+};
\path (k) -- (g) node [midway] {+};
\path (d) -- (e) node [midway] {+};
\path (o) -- (p) node [midway] {+};
\path (o) -- (e) node (x) [midway] {$\cdots$}
  child [grow=down] {
    node (y) {$O\left(\displaystyle\sum_{i = 0}^k 2^i \cdot \frac{n}{2^i}\right)$}
    edge from parent[draw=none]
  };
\path (q) -- (r) node [midway] {+};
\path (s) -- (r) node [midway] {+};
\path (s) -- (t) node [midway] {+};
\path (s) -- (l) node [midway] {=};
\path (t) -- (u) node [midway] {+};
\path (z) -- (u) node [midway] {=};
\path (j) -- (t) node [midway] {=};
\path (y) -- (x) node [midway] {$\Downarrow$};
\path (v) -- (y)
  node (w) [midway] {$O\left(\displaystyle\sum_{i = 0}^k n\right) = O(k \cdot n)$};
\path (q) -- (v) node [midway] {=};
\path (e) -- (x) node [midway] {+};
\path (o) -- (x) node [midway] {+};
\path (y) -- (w) node [midway] {$=$};
\path (v) -- (w) node [midway] {$\Leftrightarrow$};
\path (r) -- (c) node [midway] {$\cdots$};
\end{tikzpicture}
\end{document}
```

<span class="caption">
<a href="https://texample.net/tikz/examples/merge-sort-recursion-tree/">Source</a>
</span>

Note that this uses an external library which requires an internet connection to render the diagram.

## Blockquotes

**notie** supports different types of blockquotes:

```markdown
<blockquote class="definition">

A **definition** is a statement that explains the meaning of a term.

</blockquote>
```

<blockquote class="definition">

A **definition** is a statement that explains the meaning of a term.

</blockquote>

You can use the following classes to create different types of blocks:

- `definition`
- `proof`
- `equation`
- `theorem`
- `important`

```markdown
<blockquote class="proof">

A **proof** is a logical argument that demonstrates the truth of a statement.

</blockquote>
```

<blockquote class="proof">

A **proof** is a logical argument that demonstrates the truth of a statement.

</blockquote>

```markdown
<blockquote class="equation">

$$
\begin{equation}
a^2 + b^2 = c^2
\end{equation}
$$

</blockquote>
```

<blockquote class="equation">

$$
\begin{equation}
a^2 + b^2 = c^2
\end{equation}
$$

</blockquote>

```markdown
<blockquote class="theorem">

**Theorem.** This is a theorem.

</blockquote>
```

<blockquote class="theorem">

**Theorem.** This is a theorem.

</blockquote>

```markdown
<blockquote class="important">

**Important.** This is an important note.

</blockquote>
```

<blockquote class="important">

**Important.** This is an important note.

</blockquote>

## Collapsible Sections

You can create collapsible sections in your markdown file by using the following syntax:

```markdown
<details><summary>Click to expand!</summary>

This is a collapsible section. You can write markdown content here, such as:

$$
\begin{equation}
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
\end{equation}
$$

</details>
```

<details><summary>Click to expand!</summary>

This is a collapsible section. You can write markdown content here, such as:

$$
\begin{equation}
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
\end{equation}
$$

</details>

## Tables

**notie** supports tables in markdown files. You can create tables using the following syntax:

```markdown
| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Row 1    | Row 1    | Row 1    |
| Row 2    | Row 2    | Row 2    |
| Row 3    | Row 3    | Row 3    |
```

| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Row 1    | Row 1    | Row 1    |
| Row 2    | Row 2    | Row 2    |
| Row 3    | Row 3    | Row 3    |

## Images

You can include images in your markdown file using the following syntax:

```markdown
![Alt text](https://via.placeholder.com/150)
```

![Alt text](https://via.placeholder.com/150)

You can also make the images smaller by using HTML attributes:

```markdown
<img src="https://via.placeholder.com/150" alt="placeholder" style="display: block; max-height: 30%; max-width: 30%;">

<span class="caption">
This is a placeholder image.
</span>
```

<img src="https://via.placeholder.com/150" alt="placeholder" style="display: block; max-height: 30%; max-width: 30%;">

<span class="caption">
This is a placeholder image.
</span>

## Links

You can include links in your markdown file using the following syntax:

```markdown
[Click here to visit Google](https://www.google.com)
```

[Click here to visit Google](https://www.google.com)

## Footnotes

You can cite things using `[^1]` and then define the footnote at the bottom of the document like so:

I am citing something here[^1]. Something else here[^2]. Multiple things here[^3] [^4].

[^1]: This is the footnote.

[^2]: This is another footnote.

[^3]: This is yet another footnote.

[^4]: This is the last footnote.

## Request for more features

If you would like to see more features in **notie**, please feel free to open an issue on the [GitHub repository](https://github.com/branyang02/notie).
