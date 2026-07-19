/**
 * Benchmark for the real notie render pipeline.
 *
 * Measures, in order:
 *   1. processor            - MarkdownProcessor.process() (pure computation).
 *   2. SSR (renderToString) - the REAL component tree (<Notie>), i.e. the
 *      actual `components` map with StaticCodeBlock, LazyRender,
 *      EquationReference, BlockquoteReference, TOC, etc. A bare
 *      <ReactMarkdown> baseline (what this script used to measure
 *      exclusively) is kept for comparison.
 *   3. Client mount (jsdom) - createRoot().render(<Notie>) in a jsdom
 *      document, including effects: incremental section reveal, async shiki
 *      syntax highlighting, equation/blockquote numbering, TOC observers.
 *
 * Honest caveats are printed with the results: renderToString cannot await
 * async effects (shiki highlighting is excluded from SSR numbers), and jsdom
 * performs no real layout/paint, so client numbers measure JS/DOM work only.
 *
 * Usage:
 *   npm run benchmark
 *   NOTIE_BENCHMARK_MARKDOWN=path/to/note.md npm run benchmark
 */

// notie components import CSS/CSS-module files, which Node cannot parse.
// Stub out the .css loader before the component modules are (dynamically)
// imported below. This mirrors what bundlers/vitest do with CSS imports.
require.extensions[".css"] = () => {};

import fs from "node:fs";
import { performance } from "node:perf_hooks";
import React from "react";
import { renderToString } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { FullNotieConfig } from "../src/config/NotieConfig";
import { katexOptions } from "../src/utils/katexOptions";
import { MarkdownProcessor } from "../src/utils/MarkdownProcessor";

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

type ProcessedMarkdown = ReturnType<MarkdownProcessor["process"]>;

function measure<T>(label: string, fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;
    console.log(`${label}: ${duration.toFixed(1)}ms`);
    return result;
}

function renderBareMarkdown(markdownContent: string) {
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

async function poll(
    predicate: () => boolean,
    timeoutMs: number,
    intervalMs = 10,
): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (predicate()) return true;
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
    return predicate();
}

async function runServerRenderBenchmarks(
    markdown: string,
    processed: ProcessedMarkdown,
): Promise<void> {
    console.log("\n--- SSR: real component pipeline (renderToString) ---");

    const importStart = performance.now();
    const { default: Notie } = await import("../src/components/Notie");
    const { default: MarkdownRenderer } = await import(
        "../src/components/MarkdownRenderer"
    );
    console.log(
        `component module import (one-time): ${(
            performance.now() - importStart
        ).toFixed(1)}ms`,
    );

    // On the server every section is rendered (no incremental reveal), so
    // this measures the full document through the real components map,
    // including the processor run inside <Notie>.
    measure("<Notie> full document renderToString", () =>
        renderToString(React.createElement(Notie, { markdown, config })),
    );

    if (processed.markdownSections.length > 0) {
        // The client initially renders only the first two sections; this
        // approximates that first-paint workload with the real components.
        measure(
            "initial visible sections (2) renderToString [real components]",
            () =>
                renderToString(
                    React.createElement(MarkdownRenderer, {
                        markdownContent: processed.markdownContent,
                        markdownSections: processed.markdownSections.slice(
                            0,
                            2,
                        ),
                        config,
                        equationMapping: processed.equationMapping,
                        blockquoteMapping: processed.blockquoteMapping,
                    }),
                ),
        );
    }

    console.log("\n--- SSR: legacy baseline (bare ReactMarkdown) ---");
    measure("bare ReactMarkdown renderToString (no components map)", () =>
        renderBareMarkdown(processed.markdownContent),
    );
}

async function runClientMountBenchmark(
    markdown: string,
    processed: ProcessedMarkdown,
): Promise<void> {
    console.log(
        "\n--- Client mount: jsdom + createRoot (includes effects) ---",
    );

    // The first section (document title/preamble) is not wrapped in
    // <div className="sections">, so count the wrappers we expect in the DOM.
    const expectedSectionCount = (
        processed.markdownContent.match(/className="sections"/g) ?? []
    ).length;

    const { JSDOM } = await import("jsdom");
    const dom = new JSDOM(
        "<!doctype html><html><body><div id='root'></div></body></html>",
        { url: "http://localhost/", pretendToBeVisual: true },
    );
    const { window } = dom;

    // jsdom has no IntersectionObserver; report everything as visible
    // immediately so LazyRender-wrapped content is included in the numbers.
    class ImmediateIntersectionObserver {
        readonly root = null;
        readonly rootMargin = "";
        readonly thresholds: number[] = [];
        private readonly callback: IntersectionObserverCallback;
        constructor(callback: IntersectionObserverCallback) {
            this.callback = callback;
        }
        observe(target: Element) {
            queueMicrotask(() => {
                this.callback(
                    [
                        {
                            isIntersecting: true,
                            target,
                        } as IntersectionObserverEntry,
                    ],
                    this as unknown as IntersectionObserver,
                );
            });
        }
        disconnect() {}
        unobserve() {}
        takeRecords(): IntersectionObserverEntry[] {
            return [];
        }
    }

    const windowGlobals = window as unknown as Record<string, unknown>;
    windowGlobals.IntersectionObserver = ImmediateIntersectionObserver;
    const elementProto = window.HTMLElement.prototype as unknown as Record<
        string,
        unknown
    >;
    elementProto.scrollTo = () => {};
    elementProto.scrollIntoView = () => {};

    const g = globalThis as unknown as Record<string, unknown>;
    g.window = window;
    g.document = window.document;
    g.IntersectionObserver = ImmediateIntersectionObserver;
    g.HTMLElement = window.HTMLElement;
    g.Element = window.Element;
    g.Node = window.Node;
    g.getComputedStyle = window.getComputedStyle.bind(window);
    g.requestAnimationFrame = window.requestAnimationFrame.bind(window);
    g.cancelAnimationFrame = window.cancelAnimationFrame.bind(window);

    // Import react-dom/client and the component after the DOM globals exist.
    const { createRoot } = await import("react-dom/client");
    const { default: Notie } = await import("../src/components/Notie");

    const container = window.document.getElementById("root");
    if (!container) throw new Error("missing #root container");
    const root = createRoot(container);

    const sectionsIn = () =>
        window.document.querySelectorAll("div.sections").length;
    const shikiBlocksIn = () =>
        window.document.querySelectorAll("pre.shiki").length;

    const mountStart = performance.now();
    root.render(React.createElement(Notie, { markdown, config }));

    const firstContent = await poll(
        () => window.document.querySelector("h1, h2, div.sections") !== null,
        30_000,
        5,
    );
    const firstContentAt = performance.now() - mountStart;
    console.log(
        firstContent
            ? `first content (initial sections mounted): ${firstContentAt.toFixed(1)}ms`
            : "first content: TIMED OUT after 30s",
    );

    // Notie reveals sections incrementally (requestIdleCallback/setTimeout);
    // wait until every section from the processor is in the document.
    const allSections = await poll(
        () => sectionsIn() >= expectedSectionCount,
        90_000,
    );
    const allSectionsAt = performance.now() - mountStart;
    console.log(
        allSections
            ? `all ${expectedSectionCount} sections revealed (incremental reveal): ${allSectionsAt.toFixed(1)}ms`
            : `section reveal TIMED OUT: ${sectionsIn()}/${expectedSectionCount} sections after ${allSectionsAt.toFixed(1)}ms`,
    );

    // Shiki highlighting lands asynchronously per StaticCodeBlock effect.
    // Wait until the highlighted-block count is stable for a settle window.
    const settleMs = 1_000;
    const highlightDeadline = Date.now() + 60_000;
    let lastCount = shikiBlocksIn();
    let lastChange = Date.now();
    while (Date.now() < highlightDeadline) {
        await new Promise((resolve) => setTimeout(resolve, 25));
        const count = shikiBlocksIn();
        if (count !== lastCount) {
            lastCount = count;
            lastChange = Date.now();
        } else if (Date.now() - lastChange >= settleMs) {
            break;
        }
    }
    const highlightAt = performance.now() - mountStart - settleMs;
    console.log(
        lastCount > 0
            ? `shiki highlighting settled (${lastCount} blocks, incl. one-time highlighter init): ${highlightAt.toFixed(1)}ms`
            : "shiki highlighting: no static code blocks highlighted",
    );

    root.unmount();
    window.close();
}

async function main() {
    const markdown = notePath
        ? fs.readFileSync(notePath, "utf8")
        : createSyntheticMarkdown();

    console.log(notePath ? `file: ${notePath}` : "file: synthetic benchmark");
    console.log(
        `size: ${(markdown.length / 1024).toFixed(1)} KiB, lines: ${
            markdown.split("\n").length
        }`,
    );

    console.log("\n--- Markdown processing ---");
    const processed = measure("processor (MarkdownProcessor.process)", () =>
        new MarkdownProcessor(markdown, config).process(),
    );
    console.log(`sections: ${processed.markdownSections.length}`);

    await runServerRenderBenchmarks(markdown, processed);
    await runClientMountBenchmark(markdown, processed);

    console.log(`
--- Caveats (what these numbers do and do not include) ---
- SSR numbers use the real components map (StaticCodeBlock, LazyRender,
  EquationReference, BlockquoteReference, TOC), but renderToString cannot
  run effects or await promises: shiki syntax highlighting is EXCLUDED from
  SSR numbers (async, effect-driven) - SSR emits the escaped <pre> fallback.
- On the server, LazyRender renders its children eagerly and Notie renders
  every section (no incremental reveal), so SSR "full document" is an upper
  bound on markup generation, not a first-paint estimate.
- Client-mount numbers run in jsdom: they include React commit, effects,
  incremental section reveal, and async shiki highlighting, but jsdom does
  no real layout, paint, or compositing - real-browser numbers will be
  higher, and layout-bound work is invisible here.
- IntersectionObserver is stubbed to report everything visible immediately,
  so LazyRender-gated content is fully mounted; network-loaded embeds
  (TikZ/Desmos scripts, CodeMirror runners) do not load under jsdom.
- "shiki highlighting settled" includes the one-time highlighter/theme
  initialization and uses a 1s settle window (already subtracted from the
  reported number).`);

    process.exit(0);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
