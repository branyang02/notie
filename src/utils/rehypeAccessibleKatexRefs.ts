import type { Element, ElementContent, Root } from "hast";

/**
 * Href prefix produced by the `\eqref`/`\ref` KaTeX macros in
 * {@link ./katexOptions.katexOptions}. Anchors with this prefix are later
 * swapped for interactive `EquationReference` components by the markdown
 * `a` component override.
 */
const REF_HREF_PREFIX = "#pre-eqn-";

/**
 * Data attribute set on reference anchors that must stay inside KaTeX's
 * `aria-hidden` HTML subtree (mixed-math case, see below). The markdown
 * `a` component override reads the resulting `data-notie-inert-ref` prop
 * and renders the anchor with `tabIndex={-1}` so no focusable element is
 * left inside an `aria-hidden` subtree.
 */
export const INERT_REF_DATA_PROP = "data-notie-inert-ref";

function hasClass(node: Element, name: string): boolean {
    const className = node.properties?.className;
    if (Array.isArray(className)) {
        return className.includes(name);
    }
    if (typeof className === "string") {
        return className.split(/\s+/).includes(name);
    }
    return false;
}

function isElement(node: ElementContent | Root): node is Element {
    return node.type === "element";
}

function collectRefAnchors(node: Element, out: Element[]): void {
    for (const child of node.children) {
        if (!isElement(child)) continue;
        if (
            child.tagName === "a" &&
            typeof child.properties?.href === "string" &&
            child.properties.href.startsWith(REF_HREF_PREFIX)
        ) {
            out.push(child);
        }
        collectRefAnchors(child, out);
    }
}

/**
 * Whether the `.katex-html` subtree renders anything visible besides the
 * (empty) reference anchors themselves. KaTeX renders glyphs as text nodes
 * inside styled spans and radicals/stretchy delimiters as inline SVG, while
 * a standalone `\eqref`/`\ref` produces only zero-height struts and an
 * empty anchor. Any real math therefore surfaces as non-whitespace text or
 * an `svg`/`img` element.
 */
function containsVisibleMath(node: Element): boolean {
    for (const child of node.children) {
        if (child.type === "text" && child.value.trim() !== "") {
            return true;
        }
        if (!isElement(child)) continue;
        if (child.tagName === "svg" || child.tagName === "img") {
            return true;
        }
        if (containsVisibleMath(child)) {
            return true;
        }
    }
    return false;
}

/**
 * Fixes the focusable-inside-`aria-hidden` violation for equation
 * reference anchors emitted by the `\eqref`/`\ref` KaTeX macros.
 *
 * KaTeX emits every formula twice: a `.katex-mathml` half for assistive
 * technology and a `.katex-html[aria-hidden="true"]` half for visual
 * layout. The `\href`-based reference macros put a focusable `<a>` inside
 * the hidden half, so keyboard focus lands on an element hidden from the
 * accessibility tree (axe rule `aria-hidden-focus`).
 *
 * Two cases, chosen per `.katex` span *after* `rehype-katex` ran:
 *
 * 1. Standalone reference (`$\eqref{eq:x}$`, the documented Notie
 *    pattern): the hidden half contains nothing visible except the empty
 *    anchor, and the MathML half is an empty `<mrow href>` plus a TeX
 *    source `<annotation>` — neither is announced. It is therefore safe to
 *    remove `aria-hidden` from the HTML half and drop the MathML half:
 *    the reference is announced exactly once, via the `aria-label` of the
 *    interactive anchor that `EquationReference` renders in place.
 *
 * 2. Reference embedded in real math (`$x = \eqref{eq:x}$`): removing
 *    `aria-hidden` would double-announce the math (MathML + layout
 *    spans), so instead the anchor is marked with
 *    {@link INERT_REF_DATA_PROP} and rendered with `tabIndex={-1}`. The
 *    reference stays clickable with a pointer but leaves the tab order,
 *    matching the rest of the hidden subtree.
 */
export default function rehypeAccessibleKatexRefs() {
    return (tree: Root) => {
        visit(tree);
    };
}

function visit(node: Root | Element): void {
    if (isElement(node) && hasClass(node, "katex")) {
        processKatexSpan(node);
        return;
    }
    for (const child of node.children) {
        if (child.type === "element") {
            visit(child);
        }
    }
}

function processKatexSpan(node: Element): void {
    const htmlHalf = node.children.find(
        (child): child is Element =>
            isElement(child) && hasClass(child, "katex-html"),
    );
    if (!htmlHalf) return;

    const refAnchors: Element[] = [];
    collectRefAnchors(htmlHalf, refAnchors);
    if (refAnchors.length === 0) return;

    if (containsVisibleMath(htmlHalf)) {
        for (const anchor of refAnchors) {
            anchor.properties = {
                ...anchor.properties,
                // tabIndex covers renderers without an `a` component
                // override (e.g. preview cards); the data prop lets the
                // override propagate inertness onto the interactive
                // EquationReference anchor it renders instead.
                tabIndex: -1,
                dataNotieInertRef: "true",
            };
        }
        return;
    }

    if (htmlHalf.properties) {
        delete htmlHalf.properties.ariaHidden;
    }
    node.children = node.children.filter(
        (child) => !(isElement(child) && hasClass(child, "katex-mathml")),
    );
}
