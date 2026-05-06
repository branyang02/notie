import fs from "node:fs";
import { performance } from "node:perf_hooks";
import React from "react";
import { renderToString } from "react-dom/server";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { MarkdownProcessor } from "../src/utils/MarkdownProcessor";
import { FullNotieConfig } from "../src/config/NotieConfig";
import { katexOptions } from "../src/utils/katexOptions";

const notePath = process.env.NOTIE_BENCHMARK_MARKDOWN;

const config: FullNotieConfig = {
    showTableOfContents: true,
    previewEquations: true,
    previewBlockquotes: true,
    tocTitle: "Contents",
    fontSize: "1.05em",
    theme: {
        appearance: "light",
        backgroundColor: "rgb(245, 244, 239)",
        fontFamily: '"Computer Modern Serif", serif',
        customFontUrl:
            "https://cdn.jsdelivr.net/gh/bitmaks/cm-web-fonts@latest/fonts.css",
        titleColor: "#000",
        textColor: "#000",
        linkColor: "#36f",
        linkHoverColor: "#0056b3",
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

function measure<T>(label: string, fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;
    console.log(`${label}: ${duration.toFixed(1)}ms`);
    return result;
}

function renderMarkdown(markdownContent: string) {
    return renderToString(
        <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[[rehypeKatex, katexOptions], rehypeRaw, rehypeSlug]}
        >
            {markdownContent}
        </ReactMarkdown>,
    );
}

function createSyntheticMarkdown(sectionCount = 80): string {
    const sections = Array.from({ length: sectionCount }, (_, index) => {
        const sectionNumber = index + 1;
        return `## Section ${sectionNumber}

This section references $\\eqref{eq:synthetic-${sectionNumber}}$ and [Algorithm](#bqref-alg:synthetic-${sectionNumber}).

$$
\\begin{equation} \\label{eq:synthetic-${sectionNumber}}
x_${sectionNumber} = \\sum_{j=0}^{${sectionNumber}} \\gamma^j r_j
\\end{equation}
$$

<blockquote class="algorithm" id="alg:synthetic-${sectionNumber}">

Use $\\eqref{eq:synthetic-${sectionNumber}}$ as the update target.

</blockquote>

~~~tsx
export function Example${sectionNumber}() {
    return <div>Section ${sectionNumber}</div>;
}
~~~
`;
    });

    return `# Synthetic Benchmark Note

${sections.join("\n")}
`;
}

const markdown = notePath
    ? fs.readFileSync(notePath, "utf8")
    : createSyntheticMarkdown();

console.log(notePath ? `file: ${notePath}` : "file: synthetic benchmark");
console.log(
    `size: ${(markdown.length / 1024).toFixed(1)} KiB, lines: ${
        markdown.split("\n").length
    }`,
);

const processed = measure("processor", () =>
    new MarkdownProcessor(markdown, config).process(),
);

measure("full markdown renderToString", () =>
    renderMarkdown(processed.markdownContent),
);

if ("markdownSections" in processed && processed.markdownSections.length > 0) {
    measure("first section renderToString", () =>
        renderMarkdown(processed.markdownSections[0]),
    );
    measure("initial visible sections renderToString", () => {
        for (const section of processed.markdownSections.slice(0, 2)) {
            renderMarkdown(section);
        }
    });
    measure("all sections renderToString", () => {
        for (const section of processed.markdownSections) {
            renderMarkdown(section);
        }
    });
    console.log(`sections: ${processed.markdownSections.length}`);
}
