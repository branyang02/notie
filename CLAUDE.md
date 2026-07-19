# Claude Notes

Read `AGENTS.md` first for the agent workflow.

## Project Layout

- `src/` contains the `notie-markdown` library: React components, markdown processing, config, styles, utilities, services, tests, and the local dev harness under `src/dev/`.
- `demo-app/` is the separate Vite demo site that consumes the library via `notie-markdown: file:..`.

## Checks

Run the standard library checks from the repo root:

```sh
npm test
npm run lint
npm run build
```

There is no committed end-to-end test runner yet. For browser/e2e smoke checks, start a dev server and inspect it with a real browser or Playwright:

```sh
npm run dev -- --host 127.0.0.1 --port 5173
```

For the demo app:

```sh
cd demo-app
npm run dev -- --host 127.0.0.1 --port 5174
```
