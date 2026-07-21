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
 * headings). remark decodes these before the rendered heading text is
 * slugged, so the TOC must decode them too (`&nbsp;` becomes U+00A0, which
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
 * source. The renderer slugs the *rendered* heading text (after markdown and
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
            // Reference-style images render no text either: ![alt][ref],
            // collapsed ![alt][], and shortcut ![alt]. Must run before the
            // reference-link replaces below so the `!` is consumed too.
            .replace(/!\[[^\]]*\](?:\[[^\]]*\])?/g, "")
            // Reference-style links keep their text: [text][ref] and
            // collapsed [text][]. Footnote references ([^1]) are not
            // links, so ^-prefixed brackets are left untouched.
            .replace(/\[(?!\^)([^\]]*)\]\[[^\]]*\]/g, "$1")
            // Shortcut reference links: bare [text] -> text. remark only
            // resolves these when a matching [text]: definition exists in
            // the same section tree, but github-slugger strips brackets
            // from unresolved literals anyway, so the slug matches either
            // way. Footnote references ([^1]) are not links and must
            // survive untouched.
            .replace(/\[(?!\^)([^\]]*)\]/g, "$1")
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

export interface TocExtractionResult {
    /** TOC entries (level >= 2) in document order. */
    entries: TocEntry[];
    /**
     * For every input section, the ids of ALL its markdown headings
     * (including the level-1 title, which is never listed in the TOC but
     * still occupies an id) in document order. These feed the
     * rehypeHeadingIds plugin so the ids rendered into the DOM are exactly
     * the document-unique ids computed here.
     */
    sectionHeadingIds: string[][];
}

/**
 * Core TOC extraction over sections whose protected regions (code blocks
 * and HTML comments) have ALREADY been masked, so `#` lines inside them are
 * never mistaken for headings.
 *
 * A single document-scoped slugger runs across ALL sections, so headings
 * repeated in different `##` sections get unique `-1`, `-2`, ... suffixes
 * document-wide. The renderer assigns these same precomputed ids to the DOM
 * (via rehypeHeadingIds), so TOC hrefs and heading ids always agree.
 *
 * This is the single shared implementation behind both
 * `extractTableOfContents` (which masks the raw document itself) and
 * `MarkdownProcessor.process()` (which reuses its own document-level mask
 * pass instead of masking a second time).
 */
export function extractTocFromMaskedSections(
    maskedSections: readonly string[],
): TocExtractionResult {
    const entries: TocEntry[] = [];
    const sectionHeadingIds: string[][] = [];

    // Document-scoped slugger shared across every section: duplicate
    // headings get unique suffixes no matter which section they are in.
    const slugger = new GithubSlugger();

    for (const section of maskedSections) {
        const headingIds: string[] = [];
        const pattern = /^(#+) (.*)$/gm;
        let match;

        while ((match = pattern.exec(section)) !== null) {
            const level = match[1].length;
            const title = headingRenderedText(match[2]);
            const id = slugger.slug(title);

            headingIds.push(id);

            // The level-1 title is not listed in the TOC, but it still
            // consumes a slug and occupies an id in the DOM.
            if (level === 1) continue;

            entries.push({ id, level, title });
        }

        sectionHeadingIds.push(headingIds);
    }

    return { entries, sectionHeadingIds };
}

/**
 * TOC entries for already-masked text. Slugging is document-scoped: see
 * `extractTocFromMaskedSections`.
 */
export function extractTocEntriesFromMasked(maskedText: string): TocEntry[] {
    return extractTocFromMaskedSections([maskedText]).entries;
}

export function extractTableOfContents(markdownContent: string): TocEntry[] {
    // Mask code blocks and HTML comments so that `#` lines inside them are
    // never mistaken for headings.
    const { maskedText } = maskProtectedRegions(markdownContent);
    return extractTocEntriesFromMasked(maskedText);
}
