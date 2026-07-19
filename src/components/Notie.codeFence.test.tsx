import { act, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireIntersection } from "../test/setup";
import Notie from "./Notie";

// CodeMirror cannot mount in jsdom (duplicate @codemirror/state instances
// break instanceof checks), so stub the editor for execute-* fences.
vi.mock("@uiw/react-codemirror", () => ({
    __esModule: true,
    default: React.forwardRef<HTMLDivElement>(
        function CodeMirrorStub(_props, ref) {
            return <div ref={ref} data-testid="codemirror-stub" />;
        },
    ),
}));

// Keep highlighting deterministic in jsdom: StaticCodeBlock keeps its
// escaped <pre><code> fallback, CodeBlock's highlighter never resolves.
vi.mock("../utils/shikiHighlighter", async () => {
    const actual = await vi.importActual<
        typeof import("../utils/shikiHighlighter")
    >("../utils/shikiHighlighter");
    return {
        ...actual,
        highlightWithCache: () => new Promise<string>(() => {}),
        getHighlighter: () => new Promise(() => {}),
    };
});

/**
 * Regression tests for issue #88: react-markdown v9 removed the `inline`
 * prop, so block-vs-inline detection must be structural (block code is a
 * <code> inside a <pre>). A fence with no info string has no language-*
 * class and used to fall through to the plain inline <code> branch,
 * rendering unstyled next to themed StaticCodeBlocks.
 */
describe("Notie code fence routing", () => {
    it("renders a classless fence through StaticCodeBlock chrome", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

\`\`\`
plain block content
\`\`\`
`}
            />,
        );

        // StaticCodeBlock chrome: a CodeHeader with the copy button and the
        // highlighted (or escaped-fallback) body inside the code-blocks div.
        expect(
            screen.getByRole("button", { name: /copy code/i }),
        ).toBeInTheDocument();

        const body = container.querySelector("[class*='code-blocks']");
        expect(body).not.toBeNull();
        expect(body!.querySelector("pre > code")).not.toBeNull();
        expect(body!.textContent).toContain("plain block content");

        // The block must NOT render as a bare inline <code> outside the
        // StaticCodeBlock wrapper.
        const content = container.querySelector(
            "[class*='blog-content']",
        ) as HTMLElement;
        const strayCode = Array.from(content.querySelectorAll("code")).filter(
            (code) => !code.closest("[class*='code-blocks']"),
        );
        expect(strayCode).toHaveLength(0);
    });

    it("keeps inline code as a plain <code> element", () => {
        const { container } = render(
            <Notie markdown={"# Demo\n\nUse `x + 1` inline."} />,
        );

        const inlineCode = Array.from(container.querySelectorAll("code")).find(
            (code) => code.textContent === "x + 1",
        );
        expect(inlineCode).toBeDefined();
        // Inline code is not wrapped in a <pre> and gets no code block chrome.
        expect(inlineCode!.closest("pre")).toBeNull();
        expect(inlineCode!.closest("[class*='code-blocks']")).toBeNull();
        expect(
            screen.queryByRole("button", { name: /copy code/i }),
        ).not.toBeInTheDocument();
    });

    it("renders language-tagged fences through StaticCodeBlock with the language label", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

\`\`\`python
print("hi")
\`\`\`
`}
            />,
        );

        expect(
            screen.getByRole("button", { name: /copy code/i }),
        ).toBeInTheDocument();
        // CodeHeader shows the language name.
        expect(screen.getByText("python")).toBeInTheDocument();
        const body = container.querySelector("[class*='code-blocks']");
        expect(body).not.toBeNull();
        expect(body!.textContent).toContain('print("hi")');
    });

    it("still routes execute-* fences to the live CodeBlock", async () => {
        render(
            <Notie
                markdown={`# Demo

\`\`\`execute-python
print("hello")
\`\`\`
`}
            />,
        );

        // CodeBlock is wrapped in LazyRender; reveal it.
        act(() => {
            fireIntersection();
        });

        await waitFor(() => {
            expect(screen.getByText("Run Code")).toBeInTheDocument();
        });
        expect(screen.getByTestId("codemirror-stub")).toBeInTheDocument();
    });

    it("still routes component fences to custom components", () => {
        render(
            <Notie
                markdown={`# Demo

\`\`\`component
{
    componentName: "Widget"
}
\`\`\`
`}
                customComponents={{
                    Widget: () => <div data-testid="custom-widget">Widget</div>,
                }}
            />,
        );

        expect(screen.getByTestId("custom-widget")).toBeInTheDocument();
    });

    it("leaves raw HTML <pre> blocks untouched", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

<pre>raw preformatted text</pre>
`}
            />,
        );

        const rawPre = Array.from(container.querySelectorAll("pre")).find(
            (pre) => pre.textContent === "raw preformatted text",
        );
        expect(rawPre).toBeDefined();
        // No StaticCodeBlock chrome around raw HTML <pre>.
        expect(rawPre!.closest("[class*='code-blocks']")).toBeNull();
        expect(
            screen.queryByRole("button", { name: /copy code/i }),
        ).not.toBeInTheDocument();
    });
});
