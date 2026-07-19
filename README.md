# **notie**: React Markdown Note-taking Component

**notie** is a versatile note-taking solution built as a simple React component, designed to seamlessly integrate Markdown-based notes into any React application. This lightweight yet powerful tool allows you to create a customized note-taking experience that includes features like **live**, **editable code execution**, **TikZ diagrams**, and **math equations**, all within a single, intuitive interface.

## Updates

9/24/2024: Version [1.3.0](https://github.com/branyang02/notie/tree/a89c10cb3ad6201971f7e89fa24b58c3b6f96633): **notie** adds support for graphing using [desmos](https://www.desmos.com/)!

8/16/2024: Version [1.2.0](https://github.com/branyang02/notie/tree/030936ad9765931f5f061e369461996719e3fdea): **notie** has migrated to using _themes_ instead of _dark mode_. This allows for more customization options and better user experience.

8/12/2024: Version [1.1.0](https://github.com/branyang02/notie/tree/f5e2539ba395e0e2540809e11ee967af1b170436): Fully integrated automatic equation numbering and equation preview.
![reference](https://github.com/user-attachments/assets/e9f0042b-0b1b-4db7-a18e-c75525b414b1)

8/2/2024: Version 1.0.0: Initial release, published on npm registry [here](https://www.npmjs.com/package/notie-markdown).

## Getting Started

To start using **notie**, install the package via npm:

```bash
npm install notie-markdown
```

Then, import the `Notie` component in your React application:

```tsx
import { Notie } from "notie-markdown";

const markdown = `# Hello World

This is a Markdown content.`;

const Example = () => <Notie markdown={markdown} />;
```

The `Notie` component is used to render Markdown content. It accepts the following props:

### Props

| Prop               | Type                          | Description                                                                                                |
| ------------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `markdown`         | `string`                      | The Markdown content to be rendered.                                                                       |
| `config`           | `NotieConfig` (optional)      | Configuration options for Notie, including table of contents settings, font size, and theme customization. |
| `theme`            | `NotieThemes` (optional)      | Predefined theme option. Can be "default", "default dark", "Starlit Eclipse", or "Starlit Eclipse Light".  |
| `customComponents` | `CustomComponents` (optional) | Custom React components to be used for rendering specific elements in the markdown.                        |

`CustomComponents` maps a `componentName` (as referenced in ` ```component ` code blocks) to a React component. Each component receives an optional `config` prop (`CustomComponentProps`) containing the full parsed JSON object from the code block; zero-prop legacy components (`() => JSX.Element`) remain compatible.

Check out the [tutorial](https://notie-markdown.vercel.app/tutorial) for more detailed information on how to use **notie**.

### Package exports

Besides the `Notie` component, the package exports the helper functions `sanitizeUrl` (a URL transform for react-markdown that blocks dangerous URL schemes) and `extractTableOfContents` (returns the `TocEntry[]` table of contents of a markdown string), plus the types `NotieProps`, `NotieConfig`, `NotieThemes`, `Theme`, `TocEntry`, `CustomComponents`, `CustomComponentProps`, `FullNotieConfig`, and `FullTheme`.

## Features

- **Live Coding**: Use the live coding feature write and **RUN** your code snippets in your notes.
- **TikZ Support**: Use TikZ to draw diagrams in your notes.
- **Math Equations**: Write math equations using LaTeX syntax.
  - **Automatic Equation Numbering**: Automatically number equations and refer to them in your notes.
- **Blockquote References**: Label definitions, theorems, lemmas, algorithms, problems, proofs, notes, and important blocks with an `id`, then reference them anywhere in the document with hover-preview tooltips.
- **Customizable Themes**: Customize the appearance of your notes with different themes.

## Security

Please keep the following in mind when using **notie**:

- **Raw HTML is rendered by design.** notie uses [rehype-raw](https://github.com/rehypejs/rehype-raw) so that HTML embedded in your Markdown (e.g. `<img>`, `<div class="caption">`) renders as-is. This means Markdown is treated as trusted input — do **not** feed untrusted, user-supplied Markdown to the `Notie` component without sanitizing it first, as it can inject arbitrary HTML (including scripts and event handlers) into your page.
- **`execute-` code blocks send code to a remote execution endpoint.** Live-executable code blocks (e.g. ` ```execute-python `) submit the code to the code execution endpoint over the network when the user clicks Run. The endpoint can be overridden via `config.codeRunnerUrl`. Be mindful of what code is sent, and point the endpoint at infrastructure you control if you have privacy or availability requirements.
- **`tikz` blocks load a third-party engine.** TikZ diagrams are rendered by the [TikZJax](https://github.com/artisticat1/obsidian-tikzjax) engine, whose script and stylesheet are fetched at runtime from a pinned third-party URL and executed in the page.
- **`desmos` blocks load the Desmos API script.** Graphs are rendered by loading the [Desmos](https://www.desmos.com/) calculator API script from Desmos servers at runtime.

## Supported markdown notes

- **ATX headings only.** The table of contents, section splitting, and automatic heading numbering only recognize ATX headings (`#`, `##`, ...). Setext headings (underlined with `===` or `---`) still render as headings, but they are not picked up by the TOC and are not numbered.
- **Display-math environments must be `$$`-wrapped.** Environments such as `\begin{equation}`/`\begin{align}` are only recognized (rendered, numbered, and referenceable via `\eqref`) when wrapped in `$$ ... $$` delimiters.
- **Theme is prop-driven.** The appearance is controlled entirely by the `theme` prop and `config.theme`; notie does not auto-detect the user's `prefers-color-scheme` setting.

## Limitations

- **One themed `Notie` instance per page.** Theming is applied by setting CSS variables (e.g. `--blog-background-color`, `--blog-text-color`) globally on the document root (`:root`). If you render multiple `Notie` components on the same page with different `theme` or `config.theme` values, they will overwrite each other's variables and all instances end up styled by whichever one applied its theme last. Multiple instances are fine as long as they share the same theme configuration.

## Configuration reference

The `config` prop accepts a `NotieConfig` object. All fields are optional; unspecified fields fall back to the selected theme's defaults.

| Option                | Type      | Description                                                                                                                                                      |
| --------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `showTableOfContents` | `boolean` | Show or hide the table of contents sidebar.                                                                                                                      |
| `tocTitle`            | `string`  | Title displayed above the table of contents.                                                                                                                     |
| `previewEquations`    | `boolean` | Show a hover preview when referencing numbered equations.                                                                                                        |
| `previewBlockquotes`  | `boolean` | Show a hover preview when referencing labeled blockquotes.                                                                                                       |
| `fontSize`            | `string`  | Base font size for the rendered notes (any CSS `font-size` value).                                                                                               |
| `codeRunnerUrl`       | `string`  | Base URL of the code-runner service used by executable code blocks.                                                                                              |
| `desmosApiKey`        | `string`  | Desmos calculator API key used by ` ```desmos ` code blocks. Defaults to the built-in demo key, which logs a warning that it is not licensed for commercial use. |
| `theme`               | `Theme`   | Fine-grained theme overrides (see below).                                                                                                                        |

### `Theme` options

| Option                     | Type                      | Description                                                        |
| -------------------------- | ------------------------- | ------------------------------------------------------------------ |
| `appearance`               | `"light" \| "dark"`       | Base appearance the theme builds on.                               |
| `backgroundColor`          | CSS color                 | Page background color.                                             |
| `fontFamily`               | CSS font-family           | Font family for note text.                                         |
| `customFontUrl`            | `string`                  | URL of a stylesheet providing the custom font.                     |
| `titleColor`               | CSS color                 | Color of the note title.                                           |
| `textColor`                | CSS color                 | Color of body text.                                                |
| `linkColor`                | CSS color                 | Link color.                                                        |
| `linkHoverColor`           | CSS color                 | Link color on hover.                                               |
| `linkUnderline`            | `boolean`                 | Underline links.                                                   |
| `tocFontFamily`            | CSS font-family           | Font family for the table of contents.                             |
| `tocCustomFontUrl`         | `string`                  | URL of a stylesheet providing the TOC font.                        |
| `tocColor`                 | CSS color                 | TOC link color.                                                    |
| `tocHoverColor`            | CSS color                 | TOC link color on hover.                                           |
| `tocUnderline`             | `boolean`                 | Underline TOC links.                                               |
| `codeColor`                | CSS color                 | Inline code text color.                                            |
| `codeBackgroundColor`      | CSS color                 | Inline code background color.                                      |
| `codeHeaderColor`          | CSS color                 | Code block header background color.                                |
| `codeFontSize`             | CSS font-size             | Code font size.                                                    |
| `codeCopyButtonHoverColor` | CSS color                 | Copy button hover color in code blocks.                            |
| `staticCodeTheme`          | Shiki theme name          | Syntax highlighting theme for static code blocks.                  |
| `liveCodeTheme`            | Shiki theme name          | Syntax highlighting theme for live (editable) code blocks.         |
| `collapseSectionColor`     | CSS color                 | Color of the collapsible section controls.                         |
| `katexSize`                | CSS font-size             | Font size for KaTeX math.                                          |
| `tableBorderColor`         | CSS color                 | Table border color.                                                |
| `tableBackgroundColor`     | CSS color                 | Table background color.                                            |
| `captionColor`             | CSS color                 | Caption text color.                                                |
| `subtitleColor`            | CSS color                 | Subtitle text color.                                               |
| `tikZstyle`                | `"inverted" \| "default"` | Render TikZ diagrams normally or color-inverted (for dark themes). |
| `blockquoteStyle`          | `"default" \| "latex"`    | Blockquote styling; `"latex"` renders LaTeX-style theorem boxes.   |
| `numberedHeading`          | `boolean`                 | Automatically number section headings.                             |
| `tocMarker`                | `boolean`                 | Show the active-section marker in the table of contents.           |

Example:

```tsx
<Notie
  markdown={markdown}
  config={{
    showTableOfContents: true,
    tocTitle: "Contents",
    fontSize: "16px",
    theme: {
      appearance: "dark",
      blockquoteStyle: "latex",
      numberedHeading: true,
    },
  }}
/>
```

## Showcase

<img width="1129" alt="image" src="https://github.com/user-attachments/assets/593799c4-ad23-4a07-8822-08bb98a194ef">
<img width="1414" alt="image" src="https://github.com/user-attachments/assets/da36f8bd-b631-4b72-bb88-25f017e76609">
<img width="1069" alt="image" src="https://github.com/user-attachments/assets/d7e22bf0-6920-4107-bea1-567abe8cfc45">
<img width="1478" alt="image" src="https://github.com/user-attachments/assets/9bb43d55-9924-49d1-a895-5c03db1ae821">
<img width="1405" alt="image" src="https://github.com/user-attachments/assets/27127d15-a019-4f26-ad29-8eba881d7bbb">
<img width="1441" alt="image" src="https://github.com/user-attachments/assets/90ea496d-2196-4a31-8556-c819399a0932">

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

**[react-markdown](https://remarkjs.github.io/react-markdown/)**

- **Author(s)**: Espen Hovlandsdal
- **License**: MIT

**[remark-math](https://remark.js.org/)**

- **Author(s)**: Junyoung Choi
- **License**: MIT

We are thankful to all the open-source projects and their contributors for making their resources available, which have greatly facilitated the development of this project.
