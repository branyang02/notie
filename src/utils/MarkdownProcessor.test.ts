import { describe, expect, it, vi } from "vitest";
import { MarkdownProcessor } from "./MarkdownProcessor";
import { FullNotieConfig } from "../config/NotieConfig";

const config: FullNotieConfig = {
    showTableOfContents: true,
    previewEquations: true,
    previewBlockquotes: true,
    tocTitle: "Contents",
    fontSize: "1rem",
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
