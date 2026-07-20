import { slug } from "github-slugger";
import type { Element, Nodes, Root } from "hast";

/**
 * Options for {@link rehypeHeadingIds}.
 */
export interface RehypeHeadingIdsOptions {
    /**
     * Precomputed ids for this tree's markdown headings, in tree order.
     * Computed once per document by the markdown processor with a single
     * document-scoped slugger, so ids are unique across ALL section trees
     * even though each section renders as an independent ReactMarkdown
     * tree (a per-tree slugger, like rehype-slug's, restarts duplicate
     * `-1`/`-2` suffixes at every section boundary and produces duplicate
     * DOM ids when headings repeat across sections).
     */
    ids?: readonly string[];
    /**
     * Every precomputed heading id in the whole document. Used to keep
     * fallback ids (for headings the markdown scan cannot see, e.g. raw
     * HTML `<h3>` elements) from colliding with ids assigned in other
     * sections.
     */
    documentIds?: readonly string[];
}

/** True when `id` is `base` or `base-<n>` (a duplicate-suffixed slug). */
function matchesBase(id: string, base: string): boolean {
    if (id === base) return true;
    return id.startsWith(`${base}-`) && /^\d+$/.test(id.slice(base.length + 1));
}

/** Concatenates all descendant text of a hast node (textContent). */
function textContent(node: Nodes): string {
    if (node.type === "text") return node.value;
    if ("children" in node) {
        let text = "";
        for (const child of node.children) {
            text += textContent(child);
        }
        return text;
    }
    return "";
}

function isHeading(node: Element): boolean {
    return /^h[1-6]$/.test(node.tagName);
}

/** Pre-order walk over element nodes (document order, like rehype-slug). */
function visitElements(node: Nodes, visitor: (element: Element) => void) {
    if (node.type === "element") {
        visitor(node);
    }
    if ("children" in node) {
        for (const child of node.children) {
            visitElements(child, visitor);
        }
    }
}

/**
 * Assigns ids to headings like rehype-slug, but from a PRECOMPUTED list of
 * document-unique ids instead of a live per-tree slugger.
 *
 * Assignment is positional with a text-match safety check: each heading's
 * rendered text is slugged (without duplicate tracking) and the next
 * precomputed id whose base matches is consumed. Headings invisible to the
 * markdown scan (raw HTML headings via rehype-raw, headings inside
 * blockquotes) match nothing and get a locally unique fallback id instead,
 * without consuming a precomputed id; scan entries that never render as
 * headings (e.g. `##` lines inside raw HTML blocks) are skipped over
 * without derailing the ids of later headings.
 *
 * All state lives inside the transform closure, so re-rendering a single
 * section re-runs the transform from scratch and assigns byte-identical
 * ids — no cross-render slugger state to corrupt.
 *
 * Without options this behaves like per-tree rehype-slug (a fresh slugger
 * per run), which keeps standalone `MarkdownRenderer` usage working.
 */
export default function rehypeHeadingIds(options?: RehypeHeadingIdsOptions) {
    const precomputed = options?.ids ?? [];
    const documentIds = options?.documentIds ?? options?.ids ?? [];

    return function (tree: Root): void {
        // Fresh per run so a section re-render assigns identical ids.
        const occupied = new Set(documentIds);
        let cursor = 0;

        visitElements(tree, (node) => {
            if (!isHeading(node) || node.properties.id) return;

            const base = slug(textContent(node));

            let id: string | undefined;
            for (let index = cursor; index < precomputed.length; index++) {
                if (matchesBase(precomputed[index], base)) {
                    id = precomputed[index];
                    cursor = index + 1;
                    break;
                }
            }

            if (id === undefined) {
                // No matching precomputed id: assign the first suffix not
                // taken by any precomputed id (or by an earlier fallback in
                // this run). Depends only on this tree's content plus the
                // static document id list, so it is stable across renders.
                let candidate = base;
                let suffix = 1;
                while (occupied.has(candidate)) {
                    candidate = `${base}-${suffix++}`;
                }
                id = candidate;
            }

            occupied.add(id);
            node.properties.id = id;
        });
    };
}
