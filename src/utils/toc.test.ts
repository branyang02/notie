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

    it("suffixes duplicate headings within a section like rehype-slug", () => {
        const markdown = `# Title

## Section

### Setup

### Setup

### Setup
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.id)).toEqual([
            "section",
            "setup",
            "setup-1",
            "setup-2",
        ]);
    });

    it("keeps duplicate suffixes unique across section boundaries (document-scoped slugger)", () => {
        // Sections render as independent trees, but heading ids are
        // precomputed here with ONE document-scoped slugger and assigned
        // to the DOM as-is (rehypeHeadingIds), so repeated headings in
        // different `##` sections still get unique ids document-wide.
        const markdown = `# Title

## Setup

### Details

## Setup

### Details
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.id)).toEqual([
            "setup",
            "details",
            "setup-1",
            "details-1",
        ]);
    });

    it("gives repeated subsection headings in different sections unique ids (issue #94)", () => {
        // /examples/programming has "## C", "## C++", and "## C#", which
        // all slug to "c"; /examples/cso2 repeats "## Processes". Both
        // must produce unique ids document-wide.
        const markdown = `# Title

## C

## C++

## C#

## Multitasking

### Processes

## Processes
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.id)).toEqual([
            "c",
            "c-1",
            "c-2",
            "multitasking",
            "processes",
            "processes-1",
        ]);

        const ids = entries.map((entry) => entry.id);
        expect(new Set(ids).size).toBe(ids.length);
    });

    it("slugs special characters exactly like github-slugger", () => {
        const markdown = `# Title

## C & D

## It's \`code\` (v2)

## Q: what?

## A+B
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.id)).toEqual([
            "c--d",
            "its-code-v2",
            "q-what",
            "ab",
        ]);
    });

    it("decodes the &nbsp; runs inserted by numbered headings", () => {
        // MarkdownProcessor's numbered-heading mode rewrites headings to
        // "2.1&nbsp;&nbsp;&nbsp;Title"; remark decodes the entities to
        // U+00A0 before rehype-slug slugs the text, and github-slugger
        // strips U+00A0 entirely.
        const markdown = `# Title

## 1&nbsp;&nbsp;&nbsp;Section One

### 1.1&nbsp;&nbsp;&nbsp;Subsection
`;

        const entries = extractTableOfContents(markdown);

        expect(entries).toEqual([
            {
                id: "1section-one",
                level: 2,
                title: "1   Section One",
            },
            {
                id: "11subsection",
                level: 3,
                title: "1.1   Subsection",
            },
        ]);
    });

    it("strips markdown syntax that does not appear in rendered heading text", () => {
        const markdown = `# Title

## See [the docs](https://example.com) here

## **Bold** and *italic* and ~~gone~~

## Inline <em>html</em> tag
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.id)).toEqual([
            "see-the-docs-here",
            "bold-and-italic-and-gone",
            "inline-html-tag",
        ]);
        expect(entries.map((entry) => entry.title)).toEqual([
            "See the docs here",
            "Bold and italic and gone",
            "Inline html tag",
        ]);
    });

    it("strips reference-style links to their text like rendered output", () => {
        const markdown = `# Title

## See [the guide][guide] here

## Read [the spec][] now

## Try [shortcut] form

[guide]: https://example.com/guide
[the spec]: https://example.com/spec
[shortcut]: https://example.com/shortcut
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.id)).toEqual([
            "see-the-guide-here",
            "read-the-spec-now",
            "try-shortcut-form",
        ]);
        expect(entries.map((entry) => entry.title)).toEqual([
            "See the guide here",
            "Read the spec now",
            "Try shortcut form",
        ]);
    });

    it("drops reference-style images from heading text", () => {
        const markdown = `# Title

## Logo ![alt text][logo] heading

## Plain ![icon] heading

[logo]: https://example.com/logo.png
[icon]: https://example.com/icon.png
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.id)).toEqual([
            "logo--heading",
            "plain--heading",
        ]);
    });

    it("keeps footnote references in heading text intact", () => {
        // [^1] is a footnote reference, not a shortcut link; github-slugger
        // keeps the caret out but remark renders the marker, so the bracket
        // content must not be treated as link text.
        const markdown = `# Title

## Heading with footnote [^1]

[^1]: The footnote.
`;

        const entries = extractTableOfContents(markdown);

        expect(entries).toHaveLength(1);
        expect(entries[0].title).toBe("Heading with footnote [^1]");
    });

    it("counts a level-1 heading toward duplicate suffixes in its own tree", () => {
        // The `# Title` heading lives in the first section's tree, so a
        // `###` heading with the same text before the first `##` gets a
        // `-1` suffix in the DOM.
        const markdown = `# Overview

### Overview

## Rest
`;

        const entries = extractTableOfContents(markdown);

        expect(entries.map((entry) => entry.id)).toEqual([
            "overview-1",
            "rest",
        ]);
    });
});
