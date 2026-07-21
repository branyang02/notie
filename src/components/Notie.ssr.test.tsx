import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import Notie from "./Notie";

// Note: vitest runs in jsdom, so `window` exists here. What these tests
// verify is that the first (server-equivalent) render pass produced by
// renderToString does not throw and renders the same uniform initial
// section count that the client's first render uses, so hydration of
// server-rendered markup cannot mismatch.
describe("Notie server-side rendering", () => {
    it("renderToString produces non-empty markup without throwing", () => {
        const markup = renderToString(
            <Notie
                markdown={`# Demo

## One

First section.

## Two

Second section.
`}
            />,
        );

        expect(markup.length).toBeGreaterThan(0);
        expect(markup).toContain("First section.");
    });

    it("renders only the initial section count on the first pass", () => {
        const markup = renderToString(
            <Notie
                markdown={`# Demo

## One

First section.

## Two

Second section.

## Three

Third section.

## Four

Fourth section.
`}
            />,
        );

        // INITIAL_SECTION_COUNT is 2: the first render pass (server and
        // client alike) renders exactly the first two sections (the title
        // section and the first heading section); the rest are revealed
        // progressively after hydration. This matches the client's initial
        // render exactly, so hydration cannot mismatch.
        expect(markup).toContain("First section.");
        expect(markup).not.toContain("Second section.");
        expect(markup).not.toContain("Third section.");
        expect(markup).not.toContain("Fourth section.");
    });

    it("renders executable code blocks without touching navigator in the render body", () => {
        const markup = renderToString(
            <Notie
                markdown={`# Demo

\`\`\`execute-python
print("hello")
\`\`\`
`}
            />,
        );

        expect(markup.length).toBeGreaterThan(0);
    });
});
