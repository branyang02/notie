import type { Element, Root } from "hast";
import { describe, expect, it } from "vitest";
import rehypeHeadingIds from "./rehypeHeadingIds";

function heading(tag: string, text: string): Element {
    return {
        type: "element",
        tagName: tag,
        properties: {},
        children: [{ type: "text", value: text }],
    };
}

function root(...children: Element[]): Root {
    return { type: "root", children };
}

function ids(tree: Root): (string | undefined)[] {
    return tree.children
        .filter(
            (node): node is Element =>
                node.type === "element" && /^h[1-6]$/.test(node.tagName),
        )
        .map((node) => node.properties.id as string | undefined);
}

describe("rehypeHeadingIds", () => {
    it("assigns precomputed ids to headings in tree order", () => {
        const tree = root(
            heading("h2", "Setup"),
            heading("h3", "Details"),
            heading("h3", "C"),
        );

        rehypeHeadingIds({ ids: ["setup-1", "details-1", "c-2"] })(tree);

        expect(ids(tree)).toEqual(["setup-1", "details-1", "c-2"]);
    });

    it("assigns identical ids when the same tree content is re-rendered", () => {
        // Progressive reveal re-renders sections; the transform must not
        // consume shared state across runs.
        const make = () => root(heading("h2", "Setup"), heading("h3", "C"));

        const first = make();
        const second = make();
        const plugin = rehypeHeadingIds({ ids: ["setup-1", "c-3"] });
        plugin(first);
        plugin(second);

        expect(ids(second)).toEqual(ids(first));
    });

    it("skips precomputed entries whose base does not match the heading text", () => {
        // A `##` line inside a raw HTML block is visible to the markdown
        // scan but never renders as a heading; its id must not derail the
        // ids of later headings.
        const tree = root(heading("h2", "Real"), heading("h3", "Later"));

        rehypeHeadingIds({ ids: ["real", "phantom", "later"] })(tree);

        expect(ids(tree)).toEqual(["real", "later"]);
    });

    it("gives headings invisible to the markdown scan a non-colliding fallback id", () => {
        // Raw HTML headings (via rehype-raw) are not in the precomputed
        // list; they get a fallback that avoids every document id.
        const tree = root(
            heading("h3", "Extra"), // not in the scan
            heading("h2", "Setup"),
        );

        rehypeHeadingIds({
            ids: ["setup"],
            documentIds: ["setup", "extra"], // "extra" taken elsewhere
        })(tree);

        expect(ids(tree)).toEqual(["extra-1", "setup"]);
    });

    it("preserves explicit ids", () => {
        const custom = heading("h2", "Custom");
        custom.properties.id = "my-anchor";
        const tree = root(custom, heading("h2", "Setup"));

        rehypeHeadingIds({ ids: ["setup"] })(tree);

        expect(ids(tree)).toEqual(["my-anchor", "setup"]);
    });

    it("behaves like per-tree rehype-slug without options", () => {
        const tree = root(
            heading("h2", "Setup"),
            heading("h2", "Setup"),
            heading("h2", "C & D"),
        );

        rehypeHeadingIds()(tree);

        expect(ids(tree)).toEqual(["setup", "setup-1", "c--d"]);
    });
});
