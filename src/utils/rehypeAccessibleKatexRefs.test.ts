import type { Element, ElementContent, Root } from "hast";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";
import { describe, expect, it } from "vitest";
import { katexOptions } from "./katexOptions";
import rehypeAccessibleKatexRefs from "./rehypeAccessibleKatexRefs";

function renderToHast(markdown: string): Root {
    const processor = unified()
        .use(remarkParse)
        .use(remarkMath)
        .use(remarkRehype)
        .use(rehypeKatex, katexOptions)
        .use(rehypeAccessibleKatexRefs);
    return processor.runSync(processor.parse(markdown)) as Root;
}

function findAll(
    node: Root | ElementContent,
    predicate: (el: Element) => boolean,
    out: Element[] = [],
): Element[] {
    if ("children" in node) {
        for (const child of node.children) {
            if (child.type === "element") {
                if (predicate(child)) out.push(child);
                findAll(child, predicate, out);
            }
        }
    }
    return out;
}

const hasClass = (el: Element, name: string) => {
    const className = el.properties?.className;
    return Array.isArray(className) && className.includes(name);
};

describe("rehypeAccessibleKatexRefs", () => {
    it("removes aria-hidden and the MathML half for standalone references", () => {
        const tree = renderToHast("See $\\eqref{eq:first}$.");

        const hiddenSubtrees = findAll(
            tree,
            (el) => el.properties?.ariaHidden === "true",
        );
        expect(hiddenSubtrees).toHaveLength(0);

        expect(
            findAll(tree, (el) => hasClass(el, "katex-mathml")),
        ).toHaveLength(0);

        // The reference anchor itself survives for the `a` component
        // override to replace, unmarked (it will be focusable).
        const anchors = findAll(tree, (el) => el.tagName === "a");
        expect(anchors).toHaveLength(1);
        expect(anchors[0].properties?.dataNotieInertRef).toBeUndefined();
    });

    it("keeps aria-hidden but marks anchors inert for references embedded in visible math", () => {
        const tree = renderToHast("$x = 1 \\quad \\eqref{eq:first}$");

        // Visible math must keep its MathML alternative and hidden HTML
        // half, otherwise screen readers would announce the math twice.
        expect(
            findAll(tree, (el) => hasClass(el, "katex-mathml")),
        ).toHaveLength(1);
        const hiddenSubtrees = findAll(
            tree,
            (el) => el.properties?.ariaHidden === "true",
        );
        expect(hiddenSubtrees).toHaveLength(1);

        const anchors = findAll(tree, (el) => el.tagName === "a");
        expect(anchors).toHaveLength(1);
        expect(anchors[0].properties?.dataNotieInertRef).toBe("true");
        expect(anchors[0].properties?.tabIndex).toBe(-1);
    });

    it("treats bare \\ref references the same as \\eqref", () => {
        const tree = renderToHast("See $\\ref{eq:first}$.");

        expect(
            findAll(tree, (el) => el.properties?.ariaHidden === "true"),
        ).toHaveLength(0);
        expect(findAll(tree, (el) => el.tagName === "a")).toHaveLength(1);
    });

    it("leaves ordinary math without references untouched", () => {
        const tree = renderToHast("$e = mc^2$");

        expect(
            findAll(tree, (el) => el.properties?.ariaHidden === "true"),
        ).toHaveLength(1);
        expect(
            findAll(tree, (el) => hasClass(el, "katex-mathml")),
        ).toHaveLength(1);
        expect(findAll(tree, (el) => el.tagName === "a")).toHaveLength(0);
    });

    it("leaves non-reference \\href anchors untouched", () => {
        const tree = renderToHast("$\\href{https://example.com}{y}$");

        // The anchor renders visible text ("y"), so the math subtree keeps
        // its aria-hidden layout half and MathML alternative.
        expect(
            findAll(tree, (el) => el.properties?.ariaHidden === "true"),
        ).toHaveLength(1);
        const anchors = findAll(tree, (el) => el.tagName === "a");
        expect(anchors).toHaveLength(1);
        expect(anchors[0].properties?.dataNotieInertRef).toBeUndefined();
    });
});
