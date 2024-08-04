# **notie**: A Markdown-based Note-taking Component

**notie** is a markdown-based note-taking React component that allows you to write, edit, and organize your notes in a clean and intuitive interface. **notie** offers the following unique features:

1. **Live/Runnable Code Blocks**: Write and run code snippets in your notes.
2. **Math Equations**: Write math equations using LaTeX.
3. **TikZ Diagrams**: Create TikZ diagrams in your notes.
4. **Dynamic**: Try adjusting the window size to see how the notes dynamically adjust to the screen size.

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

### Importing markdown files

There are different ways to import markdown files into your React application based on your build framework. For example, in [Vite](https://vitejs.dev/), you can use the [`import.meta.glob`](https://vitejs.dev/guide/features#glob-import) function to import markdown files:

```typescript
// Vite + React + Typescript
const modules = import.meta.glob('./path/to/markdown', {
  query: '?raw',
  import: 'default',
});

const App = () => {
  const [markdownContent, setMarkdownContent] = useState<string>('');

  useEffect(() => {
    const fetchNote = async () => {
      for (const path in modules) {
        const markdown = await modules[path]();
        const rawMDString = markdown as string;
        setMarkdownContent(rawMDString);
        break;
      }
    };

    fetchNote();
  }, []);

  return <Notie markdownContent={markdownContent} />;
};
```

Checkout the [`demo-app`](https://github.com/branyang02/notie/tree/main/demo-app) directory for a complete example.

### `Notie` Props

| Prop       | Type                             | Description                                                 |
| ---------- | -------------------------------- | ----------------------------------------------------------- |
| `markdown` | `string`                         | The Markdown content to be rendered.                        |
| `darkMode` | `boolean` (optional)             | A flag to enable or disable dark mode. Defaults to `false`. |
| `style`    | `React.CSSProperties` (optional) | Inline styles to apply to the component.                    |

## Showcase

![image](https://github.com/branyang02/notie/assets/107154811/c7d2ac58-2f48-4e1f-af82-bfeec266c1f7)
![image](https://github.com/branyang02/notie/assets/107154811/17fe3a55-64b7-49a0-b3c1-80a2072b5e1c)
![image](https://github.com/branyang02/notie/assets/107154811/f0438d26-847b-4859-84f2-9a5ff93420a2)
![image](https://github.com/branyang02/notie/assets/107154811/b33df6d2-2837-44aa-8648-7b85bdbabdee)
![image](https://github.com/branyang02/notie/assets/107154811/103f8f2c-6621-4e01-9c5c-c2b8d3f5b5b8)
![image](https://github.com/branyang02/notie/assets/107154811/935ed296-2cad-4bd1-af7f-3d256a3fc54c)

## Contribution

Checkout [CONTRIBUTING.md](https://github.com/branyang02/notie/blob/main/CONTRIBUTING.md).
