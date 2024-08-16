# Contributing to **notie**

Thank you for your interest in contributing! We're committed to fostering an open and welcoming environment.

The live website for the **notie** library can be found at [https://notie-markdown.vercel.app/](https://notie-markdown.vercel.app/).

We also publish the **notie** library on npm. You can find the package at [https://www.npmjs.com/package/notie-markdown](https://www.npmjs.com/package/notie-markdown).

## Quick Start

1. **Fork and Clone**: Fork the repo on GitHub, clone your fork locally.

```bash
git clone https://github.com/[github-username]/notie.git
```

2. **Install Dependencies**: Install the project dependencies.

```bash
cd notie
npm install
```

3. **Run the Project**: Run the project locally in development mode.

```bash
npm run dev
```

## Project Structure

The main **notie** project is located in the root directory, while the **demo-app** is located in the `demo-app` directory. The **demo-app** serves as the example application for the **notie** library, and it is deployed at [https://notie-markdown.vercel.app/](https://notie-markdown.vercel.app/).

When `npm run dev` is run under the root directory, the application that is served is `src/dev/App.tsx`. This file is the main entry point for the development server. `src/dev/markdown-files` contains the markdown files that are rendered by the application. This is a minimal developing application that is used when developing the **notie** library.

When testing the **notie** library in a real-world application, you can use the `demo-app` directory. First, build the **notie** library under the root directory:

```bash
# .
npm run build
```

Then, navigate to the `demo-app` directory and run the following commands:

```bash
# .
cd demo-app
```

Install the dependencies:

```bash
# ./demo-app
npm install
```

Run the application:

```bash
# ./demo-app
npm run dev
```

## Code Style and Linting

We maintain a consistent code style across the project using ESLint and Prettier. Before pushing your changes, make sure to format your code and run linting checks in the root directory:

```bash
# Format the code
npm run format

# Run linting checks
npm run lint
```

## Commit Message

The version number of the `notie-markdown` package is influenced by the commit messages. The version bump (patch, minor, or major) is determined by specific keywords in the commit message. Please follow the guidelines below to ensure the correct version bump is triggered:

### Patch Version Bump

- **Trigger:** Commit message starts with `fix`
- **Example:**
  - `fix: correct typo in README`
  - `fix: resolve issue with Markdown rendering`

### Minor Version Bump

- **Trigger:** Commit message starts with `feat`
- **Example:**
  - `feat: add dark mode support`
  - `feat: introduce new Markdown parsing feature`

### Major Version Bump

- **Trigger:** Commit message contains `BREAKING CHANGE`
- **Example:**
  - `feat: update API structure for better performance BREAKING CHANGE`
  - `refactor: overhaul configuration handling BREAKING CHANGE`

By following these guidelines, you help ensure that versioning remains consistent and meaningful.

<blockquote class="important">

**⚠️ Important Note:**: We only publish the package to npm when changes are merged to the `main` branch. GitHub Actions automates this process (`.github/workflows/publish.yml`) and the package is automatically published to npm upon merging. The version number is automatically bumped based on the commit message, as described above.

</blockquote>

## Issue Reporting Guidelines

When reporting an issue, please include the following details:

- **Description:** A clear and concise description of the problem.
- **Steps to Reproduce:** Detailed steps to reproduce the issue.
- **Expected Behavior:** What you expected to happen.
- **Actual Behavior:** What actually happened.
- **Screenshots/Code:** Include any relevant screenshots or code snippets to help illustrate the issue.
- **Environment:** Information about your environment, including browser, OS, and Node.js version.
