import { describe, expect, it } from "vitest";
import { extractTableOfContents } from "./toc";

describe("extractTableOfContents", () => {
    it("extracts real headings and skips the level-1 title", () => {
        const markdown = `# Title

## Section One

### Subsection

## Section Two
`;

        const entries = extractTableOfContents(markdown);

        expect(entries).toEqual([
            { id: "section-one", level: 2, title: "Section One" },
            { id: "subsection", level: 3, title: "Subsection" },
            { id: "section-two", level: 2, title: "Section Two" },
        ]);
    });

    it("ignores heading-like lines inside fenced code blocks", () => {
        const markdown = `# Title

## Real Section

\`\`\`bash
# Install
## Configure
\`\`\`

## Another Section
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.title)).toEqual([
            "Real Section",
            "Another Section",
        ]);
    });

    it("ignores heading-like lines inside tilde fences, indented code, and comments", () => {
        const markdown = [
            "# Title",
            "",
            "## Real Section",
            "",
            "~~~text",
            "## Tilde Fake",
            "~~~",
            "",
            "    ## Indented Fake",
            "",
            "<!-- ## Commented Fake -->",
            "",
            "### Real Subsection",
            "",
        ].join("\n");

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.title)).toEqual([
            "Real Section",
            "Real Subsection",
        ]);
    });
});
