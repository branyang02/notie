import GithubSlugger from "github-slugger";
import { maskProtectedRegions } from "./markdownMasking";

export interface TocEntry {
    id: string;
    level: number;
    title: string;
}

/**
 * Decode the small set of HTML character references that can realistically
 * appear in heading source (notably the `&nbsp;` runs inserted by numbered
 * headings). remark decodes these before rehype-slug sees the heading text,
 * so the TOC must decode them too (`&nbsp;` becomes U+00A0, which
 * github-slugger strips rather than turning into a dash).
 */
function decodeEntities(text: string): string {
    return text
        .replace(/&nbsp;/g, " ")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&quot;/g, '"')
        .replace(/&#(\d+);/g, (_match, code) =>
            String.fromCodePoint(Number(code)),
        )
        .replace(/&#[xX]([0-9a-fA-F]+);/g, (_match, code) =>
            String.fromCodePoint(parseInt(code, 16)),
        )
        .replace(/&amp;/g, "&");
}

/**
 * Approximate the rendered text content of a heading from its raw markdown
 * source. rehype-slug slugs the *rendered* heading text (after markdown and
 * inline HTML processing), while the TOC scans raw markdown, so markdown
 * syntax that disappears from the rendered output must be stripped here:
 * links/images (keep the text), inline HTML tags, code-span backticks,
 * emphasis markers, and character references.
 */
function headingRenderedText(rawTitle: string): string {
    return decodeEntities(
        rawTitle
            .replace(/\s+#+\s*$/, "") // ATX closing sequence: "## Foo ##"
            .replace(/!\[[^\]]*\]\([^)]*\)/g, "") // images render no text
            .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1") // links -> link text
            .replace(/<[^>]*>/g, "") // inline HTML tags
            .replace(/`+/g, "") // code-span markers
            .replace(/\*+/g, "") // emphasis / strong markers
            .replace(/~~/g, "") // strikethrough markers
            // Underscore emphasis only applies at word boundaries
            // (intraword snake_case is literal), so only strip flanking
            // underscore runs.
            .replace(/(^|\s)_{1,3}(?=\S)([^_]+?)_{1,3}(?=\s|$)/g, "$1$2"),
    ).trim();
}

export function extractTableOfContents(markdownContent: string): TocEntry[] {
    const entries: TocEntry[] = [];
    const pattern = /^(#+) (.*)$/gm;
    let match;

    // Mask code blocks and HTML comments so that `#` lines inside them are
    // never mistaken for headings.
    const { maskedText } = maskProtectedRegions(markdownContent);

    // Use the same slugger as rehype-slug so TOC ids match the ids that end
    // up in the DOM, including `-1`, `-2`, ... suffixes for duplicates.
    let slugger = new GithubSlugger();

    while ((match = pattern.exec(maskedText)) !== null) {
        const level = match[1].length;
        const title = headingRenderedText(match[2]);

        // Every `##` heading starts a new markdown section, and each section
        // is rendered as its own tree; rehype-slug resets its slugger at the
        // start of every tree, so duplicate suffixes restart at each section
        // boundary. Mirror that by resetting at every level-2 heading.
        if (level === 2) {
            slugger = new GithubSlugger();
        }

        const id = slugger.slug(title);

        // The level-1 title is not listed in the TOC, but it still consumes
        // a slug in its section's tree.
        if (level === 1) continue;

        entries.push({ id, level, title });
    }

    return entries;
}
