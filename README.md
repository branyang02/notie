# **notie**: React Markdown Note-taking Component

**notie** is a versatile note-taking solution built as a simple React component, designed to seamlessly integrate Markdown-based notes into any React application. This lightweight yet powerful tool allows you to create a customized note-taking experience that includes features like **live**, **editable code execution**, **TikZ diagrams**, and **math equations**, all within a single, intuitive interface.

## Updates

8/12/2024: Version [1.1.0](https://github.com/branyang02/notie/commit/f5e2539ba395e0e2540809e11ee967af1b170436): Fully integrated automatic equation numbering and equation preview.
![reference](https://github.com/user-attachments/assets/e9f0042b-0b1b-4db7-a18e-c75525b414b1)

8/2/2024: Version 1.0.0: Initial release, published on npm registry [here](https://www.npmjs.com/package/notie-markdown).

## Getting Started

To start using **notie**, install the package via npm:

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

| Prop              | Type                             | Description                                                       |
| ----------------- | -------------------------------- | ----------------------------------------------------------------- |
| `markdown`        | `string`                         | The Markdown content to be rendered.                              |
| `darkMode`        | `boolean` (optional)             | A flag to enable or disable dark mode. Defaults to `false`.       |
| `style`           | `React.CSSProperties` (optional) | Inline styles to apply to the component.                          |
| `previewEquation` | `boolean` (optional)             | A flag to enable or disable equation preview. Defaults to `true`. |

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

Checkout a full example at `/demo-app`, or visit the live website at [https://notie-markdown.vercel.app/](https://notie-markdown.vercel.app/).

## Features

- **Live Coding**: Use the live coding feature write and **RUN** your code snippets in your notes.
- **TikZ Support**: Use TikZ to draw diagrams in your notes.
- **Math Equations**: Write math equations using LaTeX syntax.
  - **Automatic Equation Numbering**: Automatically number equations and refer to them in your notes.
- **Dark Mode**: Toggle between light and dark mode for better readability.
- **Customizable**: Customize the appearance of the notes to suit your preferences.

## Showcase

<img width="1129" alt="image" src="https://github.com/user-attachments/assets/593799c4-ad23-4a07-8822-08bb98a194ef">
<img width="1414" alt="image" src="https://github.com/user-attachments/assets/da36f8bd-b631-4b72-bb88-25f017e76609">
<img width="1069" alt="image" src="https://github.com/user-attachments/assets/d7e22bf0-6920-4107-bea1-567abe8cfc45">
<img width="1408" alt="image" src="https://github.com/user-attachments/assets/70ad3b21-b5ba-43c5-8bb8-1bf41941e54c">
<img width="1443" alt="image" src="https://github.com/user-attachments/assets/654be99b-edb0-410c-87ca-703f11b0d163">

## Features and Documentation

Checkout the [tutorial](https://notie-markdown.vercel.app/tutorial) to learn more about the features and documentation of **notie**.

## Contribution

Checkout [CONTRIBUTING.md](https://github.com/branyang02/notie/blob/main/CONTRIBUTING.md).

## Acknowledgements

This project makes use of several open-source projects and resources. We extend our gratitude to the developers and maintainers of these projects. Here is a list of them along with their respective licenses:

**[React CodeMirror](https://uiwjs.github.io/react-codemirror/)**

- **Author(s)**: uiw
- **License**: MIT

**[Bootstrap](https://getbootstrap.com/)**

- **Author(s)**: The Bootstrap Authors
- **License**: MIT

**[Evergreen UI](https://evergreen.segment.com/)**

- **Author(s)**: Segment.io, Inc.
- **License**: MIT

**[KaTeX](https://katex.org/)**

- **Author(s)**: Khan Academy
- **License**: MIT

**[react-code-blocks](https://react-code-blocks-rajinwonderland.vercel.app/?path=/story/code--default)**

- **Author(s)**: Raj K Singh
- **License**: MIT

**[react-markdown](https://remarkjs.github.io/react-markdown/)**

- **Author(s)**: Espen Hovlandsdal
- **License**: MIT

**[remark-math](https://remark.js.org/)**

- **Author(s)**: Junyoung Choi
- **License**: MIT

We are thankful to all the open-source projects and their contributors for making their resources available, which have greatly facilitated the development of this project.
