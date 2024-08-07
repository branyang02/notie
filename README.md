# **notie**: Note-taking with Markdown

**notie** is a note-taking template based on Markdown. Built with React, it provides a simple and intuitive interface to write and manage notes. The template supports **live, editable code runner**, **TikZ diagrams**, and **math equations** (via [Katex](https://katex.org/)). It is designed to be easy to use and customizable, allowing users to focus on writing and organizing their notes.

## Updates

8/2/2024: Update 1.0.0: Initial release, published on npm registry [here](https://www.npmjs.com/package/notie-markdown).

## Getting Started

**notie** is built as a React component library. To get started, install the package using npm:

```bash
npm install notie-markdown
```

Then, import the `Notie` component in your React application:

```typescript
import React from "react";
import { Notie } from "notie-markdown";
```

The `Notie` component is used to render Markdown content. It accepts the following props:

### Props

| Prop       | Type                             | Description                                                 |
| ---------- | -------------------------------- | ----------------------------------------------------------- |
| `markdown` | `string`                         | The Markdown content to be rendered.                        |
| `darkMode` | `boolean` (optional)             | A flag to enable or disable dark mode. Defaults to `false`. |
| `style`    | `React.CSSProperties` (optional) | Inline styles to apply to the component.                    |

### Example Usage

```jsx
import React from "react";
import Notie from "notie-markdown";

const Example = () => (
    <Notie
        markdown="# Hello World\nThis is a Markdown content."
        darkMode={true}
    />
);
```

Checkout a full example at `/demo-app`, or visit the live website at [https://notie-nine.vercel.app/](https://notie-nine.vercel.app/).

## Features

-   **Live Coding**: Use the live coding feature write and **RUN** your code snippets in your notes.
-   **TikZ Support**: Use TikZ to draw diagrams in your notes.
-   **Math Equations**: Write math equations using LaTeX syntax.
-   **Dark Mode**: Toggle between light and dark mode for better readability.
-   **Customizable**: Customize the appearance of the notes to suit your preferences.

## Showcase

![image](https://github.com/branyang02/notie/assets/107154811/c7d2ac58-2f48-4e1f-af82-bfeec266c1f7)
![image](https://github.com/branyang02/notie/assets/107154811/17fe3a55-64b7-49a0-b3c1-80a2072b5e1c)
![image](https://github.com/branyang02/notie/assets/107154811/f0438d26-847b-4859-84f2-9a5ff93420a2)
![image](https://github.com/branyang02/notie/assets/107154811/b33df6d2-2837-44aa-8648-7b85bdbabdee)
![image](https://github.com/branyang02/notie/assets/107154811/103f8f2c-6621-4e01-9c5c-c2b8d3f5b5b8)
![image](https://github.com/branyang02/notie/assets/107154811/935ed296-2cad-4bd1-af7f-3d256a3fc54c)

## Features and Documentation

Checkout the [tutorial](https://notie-nine.vercel.app/tutorial) to learn more about the features and documentation of **notie**.

## Contribution

Checkout [CONTRIBUTING.md](https://github.com/branyang02/notie/blob/main/CONTRIBUTING.md).

## Acknowledgements

This project makes use of several open-source projects and resources. We extend our gratitude to the developers and maintainers of these projects. Here is a list of them along with their respective licenses:

### [React CodeMirror](https://uiwjs.github.io/react-codemirror/)

-   **Author(s)**: uiw
-   **License**: MIT

### [Bootstrap](https://getbootstrap.com/)

-   **Author(s)**: The Bootstrap Authors
-   **License**: MIT

### [Evergreen UI](https://evergreen.segment.com/)

-   **Author(s)**: Segment.io, Inc.
-   **License**: MIT

### [KaTeX](https://katex.org/)

-   **Author(s)**: Khan Academy
-   **License**: MIT

### [react-code-blocks](https://react-code-blocks-rajinwonderland.vercel.app/?path=/story/code--default)

-   **Author(s)**: Raj K Singh
-   **License**: MIT

### [react-markdown](https://remarkjs.github.io/react-markdown/)

-   **Author(s)**: Espen Hovlandsdal
-   **License**: MIT

### [rehype-highlight](https://github.com/rehypejs/rehype-highlight)

-   **Author(s)**: Titus Wormer
-   **License**: MIT

### [remark-math](https://remark.js.org/)

-   **Author(s)**: Junyoung Choi
-   **License**: MIT

We are thankful to all the open-source projects and their contributors for making their resources available, which have greatly facilitated the development of this project.
