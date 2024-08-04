# **notie** Turotial

## **notie** Features

### Markdown

#### Title and headings

**notie** automatically generates the title and the table of contents based on the headings in your markdown file. You can use the following syntax to create headings:

```markdown
`# Title`

`## Heading 1`

`### Heading 2`

`#### Heading 3`

`##### Heading 4`

`###### Heading 5`
```

This generates the following:

# Title

## Heading 1

### Heading 2

#### Heading 3

##### Heading 4

###### Heading 5

#### Lists

You can create ordered and unordered lists using the following syntax:

```markdown
- Unordered list item 1
- Unordered list item 2
- Unordered list item 3
```

This generates the following:

- Unordered list item 1
- Unordered list item 2
- Unordered list item 3

```markdown
1. Ordered list item 1
2. Ordered list item 2
3. Ordered list item 3
```

This generates the following:

1. Ordered list item 1
2. Ordered list item 2
3. Ordered list item 3

#### Code blocks

**notie** supports both static code blocks, and **live**, **editable**, and **runnable** code blocks.

To create a static code block, wrap your code in triple backticks and specify the language, which creates the following:

```python
def hello_world():
    print("Hello, World!")
```

To create a live code block, use `execute-[language]` after the triple backticks, which creates the following:

```execute-python
def hello_world():
    print("Hello, World!")

hello_world()
```

```execute-c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

#### LaTeX

You can write LaTeX equations in your markdown file by wrapping the equation in double dollar signs. For example,

```markdown
$$
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$
```

This generates the following:

$$
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

You can also perform inline LaTeX equations by wrapping the equation in single dollar signs. For example, `$\sqrt{2}$` generates $\sqrt{2}$.

#### TikZ diagrams

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

#### Definitions, Proofs, Equations, Theorems, Important

You can use the `blockquote` html tag to create definitions, proofs, equations, theorems, and important notes. For example,

```markdown
<blockquote class="definition">

A **definition** is a statement that explains the meaning of a term.

</blockquote>
```

This generates the following:

<blockquote class="definition">

A **definition** is a statement that explains the meaning of a term.

</blockquote>

You can use the following classes to create different types of blocks:

- `definition`
- `proof`
- `equation`
- `theorem`
- `important`

<blockquote class="proof">

**Proof.** This is a proof.

</blockquote>

<blockquote class="equation">

$$
a^2 + b^2 = c^2
$$

</blockquote>

<blockquote class="theorem">

**Theorem.** This is a theorem.

</blockquote>

<blockquote class="important">

**Important.** This is an important note.

</blockquote>

#### Collapsible Sections

You can create collapsible sections in your markdown file by using the following syntax:

```markdown
<details><summary>Click to expand!</summary>

This is a collapsible section. You can write markdown content here, such as:

$$
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

</details>
```

This generates the following:

<details><summary>Click to expand!</summary>

This is a collapsible section. You can write markdown content here, such as:

$$
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

</details>

#### Tables

You can create tables in your markdown file by using the following syntax:

```markdown
| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Row 1    | Row 1    | Row 1    |
| Row 2    | Row 2    | Row 2    |
| Row 3    | Row 3    | Row 3    |
```

This generates the following:

| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Row 1    | Row 1    | Row 1    |
| Row 2    | Row 2    | Row 2    |
| Row 3    | Row 3    | Row 3    |

#### Images

You can include images in your markdown file by using the following syntax:

```markdown
![Alt text](https://via.placeholder.com/150)
```

This generates the following:

![Alt text](https://via.placeholder.com/150)

Or you can use raw html to include images with custom sizes:

```html
<img
  src="https://via.placeholder.com/150"
  alt="Alt text"
  style="display: block; max-height: 30%; max-width: 30%;"
/>
```

This generates the following:

<img src="https://via.placeholder.com/150" alt="Alt text" style="display: block; max-height: 30%; max-width: 30%;"/>

#### Captions

You can add captions to your images by using the following syntax:

```markdown
<span class="caption">

This is a caption

</span>
```

This generates the following:

<span class="caption">

This is a caption

</span>

#### Links

You can create links in your markdown file by using the following syntax:

```markdown
[Link text](https://www.example.com)
```

This generates the following:

[Link text](https://www.example.com)

#### Footnotes

You can cite things using `[^1]` and then define the footnote at the bottom of the document like so:

I am citing something here[^1]. Something else here[^2]. Multiple things here[^3] [^4].

[^1]: This is the footnote.
[^2]: This is another footnote.
[^3]: This is yet another footnote.
[^4]: This is the last footnote.

## Request for more features

Simply create a pull request for more features!
