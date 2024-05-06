# NOTIE Turotial

NOTIE is a simple and lightweight note-taking app that allows you to write and preview your notes in real-time. It supports Markdown, LaTeX, TikZ diagrams, and more.

## Download and Setup

1. Clone the repository

```bash
git clone https://github.com/branyang02/notie.git
```

2. Install the required dependencies

```bash
cd notie && npm install
```

3. Start the development server and preview your notes!

```bash
npm run dev
```

The terminal process should tell you to open the link, which is usually `http://localhost:3000`. Open the link in your browser to start writing notes.

## Writing Notes

Once you have cloned the `notie` repo, navigate to `notie/notes` where you can see all the markdown files. Create a new markdown file and start writing your notes!

## Best Practice

You can use **any** code editor (vim, notepad, whatever) to edie you `.md` files, however, I recommend using [Visual Studio Code](https://code.visualstudio.com/) for the best experience.

### VS Code Setup

1. Open an integrated terminal and run the commands above to start a local server.
2. Create a new markdown file in the `notes` directory.
3. Search for a "Simple Browser: Show" in the command palette and click on it, and enter the local dev link (usually `http://localhost:3000`).
4. Start writing your notes in the markdown file and see the preview in real-time.

<img width="1624" alt="image" src="https://github.com/branyang02/notie/assets/107154811/13466b3d-14e4-498d-a51a-8c377aeb8d84">

There's no need to reinvent the wheel. Use the tools that VS Code offers to make your note-taking experience more enjoyable. I personally love using GitHub Copilot to generate code snippets and Markdown Preview Enhanced to preview my notes. Search for these extensions in the VS Code marketplace and install them to enhance your note-taking experience.

## Features

### Markdown

#### Title and headings

Notie automatically generates the title and the table of contents based on the headings in your markdown file. You can use the following syntax to create headings:

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

NOTIE supports both static code blocks, and **live**, **editable**, and **runnable** code blocks.

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
\begin{tikzpicture}
    \draw (0,0) -- (2,0) -- (2,2) -- (0,2) -- cycle;
\end{tikzpicture}
```

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

Notie offers `small-table` and `xsmall-table` css classes to create smaller tables. For example,

<div class="small-table">

| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Row 1    | Row 1    | Row 1    |
| Row 2    | Row 2    | Row 2    |
| Row 3    | Row 3    | Row 3    |

</div>

<div class="xsmall-table">

| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Row 1    | Row 1    | Row 1    |
| Row 2    | Row 2    | Row 2    |
| Row 3    | Row 3    | Row 3    |

</div>

#### Images

You can include images in your markdown file by using the following syntax:

```markdown
![Alt text](https://via.placeholder.com/150)
```

This generates the following:

![Alt text](https://via.placeholder.com/150)

Or you can use raw html to include images with custom sizes:

```html
<img src="https://via.placeholder.com/250" alt="Alt text" width="250" height="250" />
```

This generates the following:

<img src="https://via.placeholder.com/250" alt="Alt text" width="250" height="250">

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
