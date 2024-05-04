# NOTIE: Your personal note taking template!

NOTIE is a note taking template based on **markdown**. It is designed to be minimalistic and easy to use.

## Features

- **Live Coding**: Use the live coding feature write and **RUN** your code snippets in your notes.
- **TikZ Support**: Use TikZ to draw diagrams in your notes.
- **Dark Mode**: Toggle between light and dark mode for better readability.

## Steps to use NOTIE

1. Clone the repository

```bash
git clone https://github.com/branyang02/notie.git
```

2. Install the required dependencies

```bash
cd notie && npm install
```

3. Start the development server and start taking notes!

```bash
npm run dev
```

## Showcase

![image](https://github.com/branyang02/notie/assets/107154811/c7d2ac58-2f48-4e1f-af82-bfeec266c1f7)
![image](https://github.com/branyang02/notie/assets/107154811/17fe3a55-64b7-49a0-b3c1-80a2072b5e1c)
![image](https://github.com/branyang02/notie/assets/107154811/f0438d26-847b-4859-84f2-9a5ff93420a2)
![image](https://github.com/branyang02/notie/assets/107154811/b33df6d2-2837-44aa-8648-7b85bdbabdee)
![image](https://github.com/branyang02/notie/assets/107154811/103f8f2c-6621-4e01-9c5c-c2b8d3f5b5b8)
![image](https://github.com/branyang02/notie/assets/107154811/935ed296-2cad-4bd1-af7f-3d256a3fc54c)

## Writing notes

Simply navigate to the `/notes` directory, create a new markdown file, and start writing your notes!

### What is Markdown?

Markdown is a lightweight markup language with plain-text-formatting syntax. Check out the [Markdown Guide](https://www.markdownguide.org/) to learn more about markdown.

### Features and Documentation

1. **Live Coding**
   Simply type your code snippet inside the `execute-[language]` tag. For example, to write a live Python code snippet, use the following syntax:
   ````markdown
   ```execute-python
   print("Hello, World!")
   ```
   ````
2. **Static Code Blocks**
   Simply type the language name after the triple backticks to create a static code block. For example, to write a static Python code snippet, use the following syntax:
   ````markdown
   ```python
   print("Hello, World!")
   ```
   ````
3. **TikZ Diagrams**
   Use the following syntax to create TikZ diagrams in your notes:
   ````markdown
   ```tikz
   [type a tikz diagram]
   ```
   ````
4. **Theorems, proofs, examples, definitions** and more.
   Use the following syntax to create theorems, proofs, examples, definitions, etc. in your notes:

   ```markdown
   <blockquote class="theorem">
   [type your theorem here]
   </blockquote>
   ```

   ```markdown
   <blockquote class="proof">
   [type your proof here]
   </blockquote>
   ```

   ```markdown
   <blockquote class="example">
   [type your example here]
   </blockquote>
   ```

   ```markdown
   <blockquote class="definition">
   [type your definition here]
   </blockquote>
   ```

   To create collapsible sections in your notes, use the following syntax:

   ```markdown
    <details open> <!-- open the section by default -->
    <summary>Section Title</summary>
            [type your content here]
    </details>
   ```

## What is NOTIE?

It's just a React app with a markdown parser and a code editor. Run it locally and start taking notes! Or, deploy it to a server and take notes from anywhere!
