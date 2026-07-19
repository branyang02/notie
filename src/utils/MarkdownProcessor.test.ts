import { describe, expect, it, vi } from "vitest";
import { MarkdownProcessor } from "./MarkdownProcessor";
import { FullNotieConfig } from "../config/NotieConfig";

const config: FullNotieConfig = {
    showTableOfContents: true,
    previewEquations: true,
    previewBlockquotes: true,
    tocTitle: "Contents",
    fontSize: "1rem",
    codeRunnerUrl: "https://api.brandonyifanyang.com",
    theme: {
        appearance: "light",
        backgroundColor: "#fff",
        fontFamily: "",
        customFontUrl: "",
        titleColor: "#000",
        textColor: "#000",
        linkColor: "#36f",
        linkHoverColor: "#0000cf",
        linkUnderline: false,
        tocFontFamily: "",
        tocCustomFontUrl: "",
        tocColor: "#000",
        tocHoverColor: "#777",
        tocUnderline: false,
        codeColor: "#000",
        codeBackgroundColor: "#fafafa",
        codeHeaderColor: "rgba(175, 184, 193, 0.2)",
        codeFontSize: "medium",
        codeCopyButtonHoverColor: "#F4F5F9",
        staticCodeTheme: "github-light",
        liveCodeTheme: "github-light",
        collapseSectionColor: "#f0f0f0",
        katexSize: "1.21rem",
        tableBorderColor: "#ddd",
        tableBackgroundColor: "#f2f2f2",
        captionColor: "#555",
        subtitleColor: "#969696",
        tikZstyle: "default",
        blockquoteStyle: "latex",
        numberedHeading: true,
        tocMarker: false,
    },
};

describe("MarkdownProcessor", () => {
    it("preserves equation and blockquote mappings while numbering headings", () => {
        const markdown = `# Title

## Section One

$$
\\begin{equation} \\label{eq:first}
x = 1
\\end{equation}
$$

<blockquote class="definition" id="def:first">
Definition body.
</blockquote>

## Section Two

See $\\eqref{eq:first}$ and [definition](#bqref-def:first).
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.markdownContent).toContain(
            "## 1&nbsp;&nbsp;&nbsp;Section One",
        );
        expect(result.markdownContent).toContain(
            "## 2&nbsp;&nbsp;&nbsp;Section Two",
        );
        expect(result.equationMapping["eq:first"]).toEqual({
            equationNumber: "1.1",
            equationString: "\n \nx = 1\n\n",
        });
        expect(result.blockquoteMapping["def:first"]).toEqual({
            blockquoteNumber: "1.1",
            blockquoteType: "definition",
            blockquoteContent: "Definition body.",
        });
    });

    it("does not scan labels inside fenced code blocks", () => {
        const errorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const markdown = `# Title

## Section

$$
\\begin{equation} \\label{eq:real}
x = 1
\\end{equation}
$$

\`\`\`tex
$$
\\begin{equation} \\label{eq:real}
not real
\\end{equation}
$$
<blockquote class="definition" id="def:not-real">Nope</blockquote>
\`\`\`
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(Object.keys(result.equationMapping)).toEqual(["eq:real"]);
        expect(result.blockquoteMapping).toEqual({});
        expect(errorSpy).not.toHaveBeenCalled();
        errorSpy.mockRestore();
    });
});

describe("MarkdownProcessor document-level masking", () => {
    it("does not split sections on '## ' lines inside fenced code blocks", () => {
        const fence = [
            "```bash",
            "## comment inside code",
            "echo hello",
            "```",
        ].join("\n");
        const markdown = `# Title

## Section One

Some prose.

${fence}

## Section Two

More prose.
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // Title section + two real sections; the '## ' line inside the
        // fence must not create a fourth section.
        expect(result.markdownSections).toHaveLength(3);
        // The fence survives intact, in a single section.
        expect(result.markdownSections[1]).toContain(fence);
        expect(result.markdownContent).toContain(fence);
        // No mapping pollution.
        expect(result.equationMapping).toEqual({});
        expect(result.blockquoteMapping).toEqual({});
    });

    it("ignores labels and blockquotes inside fenced code blocks", () => {
        const markdown = `# Title

## Section

\`\`\`tex
$$
\\begin{equation} \\label{eq:fake}
x = 1
\\end{equation}
$$
<blockquote class="theorem" id="thm:fake">Fake</blockquote>
\`\`\`
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping).toEqual({});
        expect(result.blockquoteMapping).toEqual({});
    });

    it("ignores labels inside indented code blocks", () => {
        const markdown = [
            "# Title",
            "",
            "## Section",
            "",
            "    $$",
            "    \\begin{equation} \\label{eq:indented}",
            "    x = 1",
            "    \\end{equation}",
            "    $$",
            "",
            "Prose after.",
            "",
        ].join("\n");

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping).toEqual({});
        expect(result.markdownContent).toContain(
            "    \\begin{equation} \\label{eq:indented}",
        );
    });

    it("does not let a stray ``` in prose swallow later content", () => {
        const markdown = `# Title

## Section

This paragraph mentions a stray fence marker below.

\`\`\`

$$
\\begin{equation} \\label{eq:visible}
x = 1
\\end{equation}
$$

Trailing prose stays intact.
`;

        // The stray \`\`\` opens an unterminated fence that runs to EOF, so
        // everything after it is code: the label must NOT be mapped, and
        // the raw text must survive verbatim.
        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping).toEqual({});
        expect(result.markdownContent).toContain(
            "This paragraph mentions a stray fence marker below.",
        );
        expect(result.markdownContent).toContain(
            "Trailing prose stays intact.",
        );
    });

    it("does not pair a stray ``` in prose with a later real fence", () => {
        const markdown = `# Title

## Section

Stray marker: \`\`\` appears mid-line here.

$$
\\begin{equation} \\label{eq:real}
x = 1
\\end{equation}
$$

\`\`\`tex
\\label{eq:in-code}
\`\`\`
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // The inline \`\`\` is not at line start, so it is not a fence: the
        // real equation is mapped, while the label inside the real fence
        // is not.
        expect(Object.keys(result.equationMapping)).toEqual(["eq:real"]);
        expect(result.markdownContent).toContain(
            "Stray marker: ``` appears mid-line here.",
        );
    });

    it("ignores blockquotes inside HTML comments and keeps counters intact", () => {
        const markdown = `# Title

## Section

<!-- <blockquote class="theorem" id="thm:ghost">Ghost</blockquote> -->

<blockquote class="theorem" id="thm:real">
Real theorem.
</blockquote>
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // No ghost entry, and the real theorem is numbered x.1 (the
        // commented-out blockquote must not increment the counter).
        expect(Object.keys(result.blockquoteMapping)).toEqual(["thm:real"]);
        expect(result.blockquoteMapping["thm:real"].blockquoteNumber).toBe(
            "1.1",
        );
        expect(result.markdownContent).toContain(
            '<!-- <blockquote class="theorem" id="thm:ghost">Ghost</blockquote> -->',
        );
    });

    it("maps labels in single-line $$\\begin{equation}...\\end{equation}$$ blocks", () => {
        const markdown = `# Title

## Section

$$\\begin{equation}\\label{eq:a} x = 1 \\end{equation}$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:a"]).toBeDefined();
        expect(result.equationMapping["eq:a"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:a"].equationString).toContain(
            "x = 1",
        );
    });

    it("counts single-line and multi-line equations in document order", () => {
        const markdown = `# Title

## Section

$$\\begin{equation}\\label{eq:first} x = 1 \\end{equation}$$

$$
\\begin{equation} \\label{eq:second}
y = 2
\\end{equation}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:first"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:second"].equationNumber).toBe("1.2");
    });

    it("does not number headings inside fenced code blocks", () => {
        const markdown = `# Title

## Real Heading

\`\`\`markdown
## Fake Heading
### Another Fake
\`\`\`

### Real Subheading
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.markdownContent).toContain(
            "## 1&nbsp;&nbsp;&nbsp;Real Heading",
        );
        expect(result.markdownContent).toContain(
            "### 1.1&nbsp;&nbsp;&nbsp;Real Subheading",
        );
        // Headings inside the fence stay untouched.
        expect(result.markdownContent).toContain("## Fake Heading");
        expect(result.markdownContent).toContain("### Another Fake");
        expect(result.markdownContent).not.toContain(
            "## 2&nbsp;&nbsp;&nbsp;Fake Heading",
        );
    });

    it("round-trips a complex document without losing code or comments", () => {
        const backtickFence = [
            "```python",
            "## not a heading",
            "def f():",
            '    return "$$\\\\label{eq:code}"',
            "```",
        ].join("\n");
        const tildeFence = [
            "~~~text",
            "### also not a heading",
            '<blockquote class="definition" id="def:code">nope</blockquote>',
            "~~~",
        ].join("\n");
        const indented = [
            "    indented code line one",
            "    \\label{eq:indented-code}",
        ].join("\n");
        const comment = "<!-- hidden ## heading and \\label{eq:comment} -->";

        const markdown = `# Title

## Alpha

${backtickFence}

$$
\\begin{equation} \\label{eq:alpha}
a = 1
\\end{equation}
$$

${comment}

## Beta

${tildeFence}

${indented}

<blockquote class="theorem" id="thm:beta">
Beta theorem.
</blockquote>
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // All protected content is restored verbatim.
        expect(result.markdownContent).toContain(backtickFence);
        expect(result.markdownContent).toContain(tildeFence);
        expect(result.markdownContent).toContain(indented);
        expect(result.markdownContent).toContain(comment);

        // Only real constructs are mapped.
        expect(Object.keys(result.equationMapping)).toEqual(["eq:alpha"]);
        expect(result.equationMapping["eq:alpha"].equationNumber).toBe("1.1");
        expect(Object.keys(result.blockquoteMapping)).toEqual(["thm:beta"]);
        expect(result.blockquoteMapping["thm:beta"].blockquoteNumber).toBe(
            "2.1",
        );

        // Section structure: title + Alpha + Beta.
        expect(result.markdownSections).toHaveLength(3);
        expect(result.markdownContent).toBe(result.markdownSections.join(""));
    });
});
