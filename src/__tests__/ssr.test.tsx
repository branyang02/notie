import { describe, expect, it } from "vitest";
import { renderToString } from "react-dom/server";
import { Notie } from "../index";

const sampleMarkdown = `# Title

Some paragraph with inline math $a^2 + b^2 = c^2$ and a link [example](https://example.com).

## Section A

A display equation:

$$
\\begin{equation}
E = mc^2 \\label{eq:einstein}
\\end{equation}
$$

A code block:

\`\`\`python
def hello():
    print("hi")
\`\`\`

A blockquote:

> "Quotation here."

## Section B

> <span class="theorem">A theorem statement.</span>
`;

const executeMarkdown = `# Execute block test

\`\`\`execute-python
print("hello")
\`\`\`
`;

describe("Notie SSR", () => {
    it("renders to string without throwing for general markdown", () => {
        const html = renderToString(<Notie markdown={sampleMarkdown} />);
        expect(html).toBeTypeOf("string");
        expect(html.length).toBeGreaterThan(0);
    });

    it("includes the title heading in SSR output", () => {
        const html = renderToString(<Notie markdown={sampleMarkdown} />);
        expect(html).toContain("Title");
    });

    it("includes a katex-rendered span for inline math", () => {
        const html = renderToString(<Notie markdown={sampleMarkdown} />);
        expect(html).toContain("katex");
    });

    it("includes the display equation tex source", () => {
        const html = renderToString(<Notie markdown={sampleMarkdown} />);
        expect(html).toContain("E = mc^2");
    });

    it("includes the code block content", () => {
        const html = renderToString(<Notie markdown={sampleMarkdown} />);
        expect(html).toContain("def hello");
    });

    it("includes the section headings", () => {
        const html = renderToString(<Notie markdown={sampleMarkdown} />);
        expect(html).toContain("Section A");
        expect(html).toContain("Section B");
    });

    it("renders a markdown containing an execute-* block without throwing", () => {
        // This is the path that previously accessed `navigator.platform`
        // unconditionally during render, which throws under Node.
        const html = renderToString(<Notie markdown={executeMarkdown} />);
        expect(html).toBeTypeOf("string");
        expect(html).toContain("Execute block test");
    });

    it("renders without errors when only markdown content is provided", () => {
        // smallest possible input
        const html = renderToString(<Notie markdown="hello world" />);
        expect(html).toContain("hello world");
    });
});
