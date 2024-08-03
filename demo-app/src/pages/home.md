# **notie**: Your personal note taking template!

**notie** is a markdown-based note-taking template that allows you to write, edit, and organize your notes in a clean and intuitive interface. **notie** offers the following unique features:

1. **Live/Runnable Code Blocks**: Write and run code snippets in your notes.
2. **Math Equations**: Write math equations using LaTeX.
3. **TikZ Diagrams**: Create TikZ diagrams in your notes.

## Getting Started

**notie** is built as a React component library. To get started, install the package using npm:

```bash
npm install notie-markdown
```

Then, import the `Notie` component in your React application:

```typescript
import { Notie } from 'notie-markdown';

const markdownContent = `# Hello, World!`;

const App = () => {
  return <Notie markdownContent={markdownContent} />;
};

export default App;
```

## Showcase

![image](https://github.com/branyang02/notie/assets/107154811/c7d2ac58-2f48-4e1f-af82-bfeec266c1f7)
![image](https://github.com/branyang02/notie/assets/107154811/17fe3a55-64b7-49a0-b3c1-80a2072b5e1c)
![image](https://github.com/branyang02/notie/assets/107154811/f0438d26-847b-4859-84f2-9a5ff93420a2)
![image](https://github.com/branyang02/notie/assets/107154811/b33df6d2-2837-44aa-8648-7b85bdbabdee)
![image](https://github.com/branyang02/notie/assets/107154811/103f8f2c-6621-4e01-9c5c-c2b8d3f5b5b8)
![image](https://github.com/branyang02/notie/assets/107154811/935ed296-2cad-4bd1-af7f-3d256a3fc54c)

## Contribution

Checkout [CONTRIBUTING.md](https://github.com/branyang02/notie/blob/main/CONTRIBUTING.md).

## Acknowledgements

This project makes use of several open-source projects and resources. We extend our gratitude to the developers and maintainers of these projects. Here is a list of them along with their respective licenses:

- [React CodeMirror](https://uiwjs.github.io/react-codemirror/)

  - **Author(s)**: uiw
  - **License**: MIT

- [Bootstrap](https://getbootstrap.com/)

  - **Author(s)**: The Bootstrap Authors
  - **License**: MIT

- [Evergreen UI](https://evergreen.segment.com/)

  - **Author(s)**: Segment.io, Inc.
  - **License**: MIT

- [KaTeX](https://katex.org/)

  - **Author(s)**: Khan Academy
  - **License**: MIT

- [react-code-blocks](https://react-code-blocks-rajinwonderland.vercel.app/?path=/story/code--default)

  - **Author(s)**: Raj K Singh
  - **License**: MIT

- [react-markdown](https://remarkjs.github.io/react-markdown/)

  - **Author(s)**: Espen Hovlandsdal
  - **License**: MIT

- [rehype-highlight](https://github.com/rehypejs/rehype-highlight)

  - **Author(s)**: Titus Wormer
  - **License**: MIT

- [remark-math](https://remark.js.org/)
  - **Author(s)**: Junyoung Choi
  - **License**: MIT

We are thankful to all the open-source projects and their contributors for making their resources available, which have greatly facilitated the development of this project.
