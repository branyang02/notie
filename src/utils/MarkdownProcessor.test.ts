import { describe, expect, it, vi } from "vitest";
import { MarkdownProcessor } from "./MarkdownProcessor";
import { extractTableOfContents } from "./toc";
import { FullNotieConfig } from "../config/NotieConfig";
import { DEFAULT_DESMOS_API_KEY } from "../components/DesmosGraph";

const config: FullNotieConfig = {
    showTableOfContents: true,
    previewEquations: true,
    previewBlockquotes: true,
    tocTitle: "Contents",
    fontSize: "1rem",
    codeRunnerUrl: "https://api.brandonyifanyang.com",
    desmosApiKey: DEFAULT_DESMOS_API_KEY,
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

    it("maps proof, note, and important blockquotes with independent counters", () => {
        const markdown = `# Title

## Section One

<blockquote class="proof" id="proof:first">
Proof body.
</blockquote>

<blockquote class="note" id="note:first">
Note body.
</blockquote>

<blockquote class="proof" id="proof:second">
Second proof body.
</blockquote>

<blockquote class="important" id="imp:first">
Important body.
</blockquote>

<blockquote class="theorem" id="thm:first">
Theorem body.
</blockquote>

<blockquote class="lemma" id="lem:first">
Lemma body.
</blockquote>

See [proof](#bqref-proof:first) and [note](#bqref-note:first).
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // proof/note/important each keep their own counter...
        expect(result.blockquoteMapping["proof:first"]).toEqual({
            blockquoteNumber: "1.1",
            blockquoteType: "proof",
            blockquoteContent: "Proof body.",
        });
        expect(result.blockquoteMapping["proof:second"]).toEqual({
            blockquoteNumber: "1.2",
            blockquoteType: "proof",
            blockquoteContent: "Second proof body.",
        });
        expect(result.blockquoteMapping["note:first"]).toEqual({
            blockquoteNumber: "1.1",
            blockquoteType: "note",
            blockquoteContent: "Note body.",
        });
        expect(result.blockquoteMapping["imp:first"]).toEqual({
            blockquoteNumber: "1.1",
            blockquoteType: "important",
            blockquoteContent: "Important body.",
        });

        // ...while theorem and lemma still share a counter.
        expect(result.blockquoteMapping["thm:first"].blockquoteNumber).toBe(
            "1.1",
        );
        expect(result.blockquoteMapping["lem:first"].blockquoteNumber).toBe(
            "1.2",
        );
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

    it("maps labels inside list-continuation display equations", () => {
        const markdown = [
            "# Title",
            "",
            "## Section",
            "",
            "1. **Case One**: A displayed equation belongs to this list item:",
            "",
            "    $$",
            "    \\begin{equation} \\label{eq:list-equation}",
            "    x = 1",
            "    \\end{equation}",
            "    $$",
            "",
            "    See $\\eqref{eq:list-equation}$ from inside the same item.",
            "",
            "See it again from outside: $\\eqref{eq:list-equation}$.",
            "",
        ].join("\n");

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:list-equation"]).toBeDefined();
        expect(result.equationMapping["eq:list-equation"].equationNumber).toBe(
            "1.1",
        );
        expect(
            result.equationMapping["eq:list-equation"].equationString,
        ).toContain("x = 1");
    });

    it("maps list-continuation equations after lazy continuation text", () => {
        const markdown = [
            "# Title",
            "",
            "## Section",
            "",
            "3. **Case Three**: A list item begins here,",
            "and this unindented line lazily continues the same item:",
            "",
            "    $$",
            "    \\begin{align}",
            "    y &= 1 \\label{eq:lazy-list-align} \\\\",
            "    \\end{align}",
            "    $$",
            "",
            "The reference $\\eqref{eq:lazy-list-align}$ should resolve.",
            "",
        ].join("\n");

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:lazy-list-align"]).toBeDefined();
        expect(
            result.equationMapping["eq:lazy-list-align"].equationNumber,
        ).toBe("1.1");
    });

    it("still ignores labels inside code blocks nested in list items", () => {
        const markdown = [
            "# Title",
            "",
            "## Section",
            "",
            "1. Item with an indented code block:",
            "",
            "       $$",
            "       \\begin{equation} \\label{eq:list-code}",
            "       x = 1",
            "       \\end{equation}",
            "       $$",
            "",
        ].join("\n");

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping).toEqual({});
        expect(result.markdownContent).toContain(
            "       \\begin{equation} \\label{eq:list-code}",
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

    it("normalizes single-line equation environments to multi-line display math", () => {
        const markdown = `# Title

## Section

$$\\begin{equation}\\label{eq:x} y = 1 \\end{equation}$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // The label is mapped...
        expect(result.equationMapping["eq:x"]).toBeDefined();
        expect(result.equationMapping["eq:x"].equationNumber).toBe("1.1");
        // ...and the returned content contains the canonical multi-line
        // form (so remark-math parses it as display math), not the
        // single-line form (which remark-math treats as inline math and
        // KaTeX rejects with "{equation} can be used only in display
        // mode").
        expect(result.markdownContent).toContain(
            "$$\n\\begin{equation}\\label{eq:x} y = 1 \\end{equation}\n$$",
        );
        expect(result.markdownContent).not.toContain(
            "$$\\begin{equation}\\label{eq:x} y = 1 \\end{equation}$$",
        );
        expect(result.markdownContent).toBe(result.markdownSections.join(""));
    });

    it("normalizes single-line align environments to multi-line display math", () => {
        const markdown = `# Title

## Section

$$\\begin{align}a &= 1 \\label{eq:a1} \\\\ b &= 2 \\label{eq:a2}\\end{align}$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // Each align row is numbered exactly once, in order.
        expect(result.equationMapping["eq:a1"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:a2"].equationNumber).toBe("1.2");
        // The delimiters and rows land on their own lines.
        expect(result.markdownContent).toContain(
            "$$\n\\begin{align}\na &= 1 \\label{eq:a1} \\\\\nb &= 2 \\label{eq:a2}\n\\end{align}\n$$",
        );
        expect(result.markdownContent).not.toContain("$$\\begin{align}");
    });

    it("does not normalize single-line environments inside fenced code blocks", () => {
        const singleLine =
            "$$\\begin{equation}\\label{eq:in-code} y = 1 \\end{equation}$$";
        const markdown = `# Title

## Section

\`\`\`tex
${singleLine}
\`\`\`
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping).toEqual({});
        // The code block content is restored verbatim, still single-line.
        expect(result.markdownContent).toContain(singleLine);
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

describe("MarkdownProcessor align-environment numbering", () => {
    it("skips \\nonumber lines so later labels match KaTeX numbering", () => {
        const markdown = `# Title

## Section

$$
\\begin{align}
a &= 1 \\nonumber \\\\
b &= 2 \\label{eq:c}
\\end{align}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // KaTeX gives the \nonumber line no number, so eq:c is the first
        // numbered equation: 1.1, not 1.2.
        expect(result.equationMapping["eq:c"].equationNumber).toBe("1.1");
    });

    it("numbers labeled lines around a \\nonumber line consecutively", () => {
        const markdown = `# Title

## Section

$$
\\begin{align}
a &= 1 \\label{eq:a} \\\\
b &= 2 \\nonumber \\\\
c &= 3 \\label{eq:b}
\\end{align}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:a"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:b"].equationNumber).toBe("1.2");
    });

    it("treats \\notag the same as \\nonumber", () => {
        const markdown = `# Title

## Section

$$
\\begin{align}
a &= 1 \\notag \\\\
b &= 2 \\label{eq:c}
\\end{align}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:c"].equationNumber).toBe("1.1");
    });

    it("warns and skips mapping when \\label and \\nonumber share a line", () => {
        const errorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const markdown = `# Title

## Section

$$
\\begin{align}
a &= 1 \\label{eq:ghost} \\nonumber \\\\
b &= 2 \\label{eq:real}
\\end{align}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:ghost"]).toBeUndefined();
        expect(result.equationMapping["eq:real"].equationNumber).toBe("1.1");
        expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining("eq:ghost"),
        );
        errorSpy.mockRestore();
    });

    it("counts a nested multi-line block as a single equation line", () => {
        const markdown = `# Title

## Section

$$
\\begin{align}
f(x) &= \\begin{cases}
1 & x > 0 \\\\
0 & x \\le 0
\\end{cases} \\label{eq:cases} \\\\
g(x) &= 2 \\label{eq:after}
\\end{align}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // The whole cases block is one align row: eq:cases is 1.1 and the
        // following row is 1.2 (existing nested-block behavior preserved).
        expect(result.equationMapping["eq:cases"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:after"].equationNumber).toBe("1.2");
    });

    it("warns and skips mapping for an equation environment with \\label and \\nonumber", () => {
        const errorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const markdown = `# Title

## Section

$$
\\begin{equation} \\label{eq:ghost} \\nonumber
a = 1
\\end{equation}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:ghost"]).toBeUndefined();
        expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining("eq:ghost"),
        );
        errorSpy.mockRestore();
    });

    it("does not let a \\nonumber equation environment consume a number", () => {
        const markdown = `# Title

## Section

$$
\\begin{equation} \\nonumber
a = 1
\\end{equation}
$$

$$
\\begin{equation} \\label{eq:real}
b = 2
\\end{equation}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // KaTeX gives the \nonumber equation no number, so eq:real is the
        // first numbered equation: 1.1, not 1.2.
        expect(result.equationMapping["eq:real"].equationNumber).toBe("1.1");
    });

    it("treats \\notag in an equation environment the same as \\nonumber", () => {
        const markdown = `# Title

## Section

$$
\\begin{equation} \\notag
a = 1
\\end{equation}
$$

$$
\\begin{equation} \\label{eq:real}
b = 2
\\end{equation}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:real"].equationNumber).toBe("1.1");
    });

    it("skips a nested multi-line block marked \\nonumber", () => {
        const markdown = `# Title

## Section

$$
\\begin{align}
f(x) &= \\begin{cases}
1 & x > 0 \\\\
0 & x \\le 0
\\end{cases} \\nonumber \\\\
g(x) &= 2 \\label{eq:after}
\\end{align}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // The unnumbered cases block consumes no number, so eq:after is 1.1.
        expect(result.equationMapping["eq:after"].equationNumber).toBe("1.1");
    });
});

describe("MarkdownProcessor gather/alignat environments", () => {
    it("maps labels in gather environments per row", () => {
        const markdown = `# Title

## Section

$$
\\begin{gather}
a = 1 \\\\
b = 2 \\label{eq:gather-second}
\\end{gather}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // KaTeX numbers each gather row, so the labeled second row is 1.2.
        expect(result.equationMapping["eq:gather-second"].equationNumber).toBe(
            "1.2",
        );
    });

    it("keeps an equation after a gather in sync with KaTeX numbering", () => {
        const markdown = `# Title

## Section

$$
\\begin{gather}
a = 1 \\label{eq:g1} \\\\
b = 2
\\end{gather}
$$

$$
\\begin{equation} \\label{eq:after-gather}
c = 3
\\end{equation}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // The gather consumes numbers 1.1 and 1.2 (KaTeX numbers every
        // gather row), so the following equation is 1.3, not 1.2.
        expect(result.equationMapping["eq:g1"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:after-gather"].equationNumber).toBe(
            "1.3",
        );
    });

    it("honors \\nonumber in gather rows", () => {
        const markdown = `# Title

## Section

$$
\\begin{gather}
a = 1 \\nonumber \\\\
b = 2 \\label{eq:gather-real}
\\end{gather}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:gather-real"].equationNumber).toBe(
            "1.1",
        );
    });

    it("maps labels in alignat environments with an argument", () => {
        const markdown = `# Title

## Section

$$
\\begin{alignat}{2}
a &= 1 &\\quad b &= 2 \\label{eq:aa1} \\\\
c &= 3 &\\quad d &= 4 \\label{eq:aa2}
\\end{alignat}
$$

$$
\\begin{equation} \\label{eq:after-alignat}
e = 5
\\end{equation}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:aa1"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:aa2"].equationNumber).toBe("1.2");
        expect(result.equationMapping["eq:after-alignat"].equationNumber).toBe(
            "1.3",
        );
    });

    it("normalizes single-line gather environments to multi-line display math", () => {
        const markdown = `# Title

## Section

$$\\begin{gather}a = 1 \\\\ b = 2 \\label{eq:sg}\\end{gather}$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:sg"].equationNumber).toBe("1.2");
        expect(result.markdownContent).toContain(
            "$$\n\\begin{gather}\na = 1 \\\\\nb = 2 \\label{eq:sg}\n\\end{gather}\n$$",
        );
        expect(result.markdownContent).not.toContain("$$\\begin{gather}");
    });

    it("normalizes single-line alignat environments keeping the argument", () => {
        const markdown = `# Title

## Section

$$\\begin{alignat}{2}a &= 1 \\label{eq:sa}\\end{alignat}$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:sa"].equationNumber).toBe("1.1");
        expect(result.markdownContent).toContain(
            "$$\n\\begin{alignat}{2}\na &= 1 \\label{eq:sa}\n\\end{alignat}\n$$",
        );
        expect(result.markdownContent).not.toContain("$$\\begin{alignat}");
    });

    it("counts a nested block inside a gather row once", () => {
        const markdown = `# Title

## Section

$$
\\begin{gather}
f(x) = \\begin{cases}
1 & x > 0 \\\\
0 & x \\le 0
\\end{cases} \\label{eq:gcases} \\\\
g(x) = 2 \\label{eq:gafter}
\\end{gather}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping["eq:gcases"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:gafter"].equationNumber).toBe("1.2");
    });
});

describe("MarkdownProcessor title-section policy", () => {
    it("warns and skips mapping for a labeled equation in the title section", () => {
        const errorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const markdown = `# Title

Some intro text.

$$
\\begin{equation} \\label{eq:title}
x = 1
\\end{equation}
$$

## Section One

$$
\\begin{equation} \\label{eq:real}
y = 2
\\end{equation}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // No broken 0.x entry is created for the title-section label...
        expect(result.equationMapping["eq:title"]).toBeUndefined();
        // ...a console.error names the label and explains the policy...
        expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining('"eq:title"'),
        );
        expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining("title section"),
        );
        // ...and section 1 numbering is unaffected.
        expect(result.equationMapping["eq:real"].equationNumber).toBe("1.1");
        errorSpy.mockRestore();
    });

    it("warns for labeled gather rows in the title section too", () => {
        const errorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const markdown = `# Title

$$
\\begin{gather}
a = 1 \\label{eq:tg1} \\\\
b = 2 \\label{eq:tg2}
\\end{gather}
$$

## Section
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping).toEqual({});
        expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining('"eq:tg1"'),
        );
        expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining('"eq:tg2"'),
        );
        errorSpy.mockRestore();
    });

    it("warns and skips mapping for a labeled blockquote in the title section", () => {
        const errorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const markdown = `# Title

<blockquote class="definition" id="def:title">
Title definition.
</blockquote>

## Section One

<blockquote class="definition" id="def:real">
Real definition.
</blockquote>
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.blockquoteMapping["def:title"]).toBeUndefined();
        expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining('"def:title"'),
        );
        expect(result.blockquoteMapping["def:real"].blockquoteNumber).toBe(
            "1.1",
        );
        errorSpy.mockRestore();
    });

    it("does not warn for unlabeled title-section equations", () => {
        const errorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const markdown = `# Title

$$
\\begin{equation}
x = 1
\\end{equation}
$$

## Section
`;

        const result = new MarkdownProcessor(markdown, config).process();

        expect(result.equationMapping).toEqual({});
        expect(errorSpy).not.toHaveBeenCalled();
        errorSpy.mockRestore();
    });
});

describe("MarkdownProcessor single-pass TOC extraction", () => {
    const plainConfig: FullNotieConfig = {
        ...config,
        theme: { ...config.theme, numberedHeading: false },
    };

    // A fixture exercising every tricky path the two implementations must
    // agree on: duplicate headings (within and across sections), a level-1
    // title consuming a slug in its tree, special characters, markdown
    // syntax stripping, inline and reference-style links, fake headings in
    // fences/indented code/comments, and ATX closing sequences.
    const trickyMarkdown = `# Overview

### Overview

## Setup

### Setup

### Setup

## Setup

### C & D

### It's \`code\` (v2)

### See [the docs](https://example.com) here

### See [the guide][guide] too

### Read [the spec][] now

### Try [shortcut] form

### **Bold** and _flanked_ text

### Closed heading ##

\`\`\`bash
## fenced fake
\`\`\`

    ## indented fake

<!-- ## commented fake -->

[guide]: https://example.com/guide
[the spec]: https://example.com/spec
[shortcut]: https://example.com/shortcut
`;

    it("returns tocEntries identical to extractTableOfContents (plain headings)", () => {
        const result = new MarkdownProcessor(
            trickyMarkdown,
            plainConfig,
        ).process();

        expect(result.tocEntries).toEqual(
            extractTableOfContents(result.markdownContent),
        );
        expect(result.tocEntries.length).toBeGreaterThan(0);
    });

    it("returns tocEntries identical to extractTableOfContents (numbered headings)", () => {
        const result = new MarkdownProcessor(trickyMarkdown, config).process();

        expect(result.tocEntries).toEqual(
            extractTableOfContents(result.markdownContent),
        );
        // Numbered ids include the heading number with the &nbsp; run
        // stripped, matching what rehype-slug produces in the DOM (the
        // title-section "### Overview" is numbered 0.1).
        expect(result.tocEntries[0].id).toBe("01overview");
    });

    it("computes reference-style-link heading slugs like rehype-slug", () => {
        const result = new MarkdownProcessor(
            `# Title

## See [the guide][guide] here

## Read [the spec][] now

[guide]: https://example.com/guide
[the spec]: https://example.com/spec
`,
            plainConfig,
        ).process();

        expect(result.tocEntries.map((entry) => entry.id)).toEqual([
            "see-the-guide-here",
            "read-the-spec-now",
        ]);
    });

    it("never lists fake headings from code or comments in tocEntries", () => {
        const result = new MarkdownProcessor(
            trickyMarkdown,
            plainConfig,
        ).process();

        const titles = result.tocEntries.map((entry) => entry.title);
        expect(titles).not.toContain("fenced fake");
        expect(titles).not.toContain("indented fake");
        expect(titles).not.toContain("commented fake");
    });
});

describe("MarkdownProcessor masking-token collision", () => {
    it("round-trips a document containing the literal mask token", () => {
        // The document contains the exact placeholder shape the masker
        // would generate (NOTIEMASK0NOTIEMASK), forcing the token base to
        // grow (base += "X") so tokens cannot collide with content.
        const literal = "NOTIEMASK0NOTIEMASK";
        const fence = ["```python", "## fake heading in code", "```"].join(
            "\n",
        );
        const markdown = `# Title

## Section

Literal token in prose: ${literal} stays byte-identical.

${fence}

$$
\\begin{equation} \\label{eq:collide}
x = ${literal}
\\end{equation}
$$
`;

        const result = new MarkdownProcessor(markdown, config).process();

        // Byte-identical round-trip of the literal, in prose and math.
        expect(result.markdownContent).toContain(
            `Literal token in prose: ${literal} stays byte-identical.`,
        );
        expect(result.markdownContent).toContain(`x = ${literal}`);
        // The masking still worked: the fence is restored verbatim and its
        // fake heading was neither numbered nor listed in the TOC.
        expect(result.markdownContent).toContain(fence);
        expect(result.markdownContent).not.toContain(
            "&nbsp;&nbsp;&nbsp;fake heading in code",
        );
        expect(result.tocEntries.map((entry) => entry.title)).not.toContain(
            "fake heading in code",
        );
        // The equation label is mapped and its stored string round-trips
        // the literal too.
        expect(result.equationMapping["eq:collide"].equationNumber).toBe("1.1");
        expect(result.equationMapping["eq:collide"].equationString).toContain(
            literal,
        );
        // No grown token base (NOTIEMASKX...) leaks into the output.
        expect(result.markdownContent).not.toMatch(/NOTIEMASKX+\d+/);
    });
});
