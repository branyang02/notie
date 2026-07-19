import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BlockquoteMapping, EquationMapping } from "../utils/utils";
import BlockquoteReference from "./BlockquoteReference";

const blockquoteMapping: BlockquoteMapping = {
    "def:first": {
        blockquoteNumber: "1.1",
        blockquoteType: "definition",
        blockquoteContent:
            "A reusable definition with \\label{def:first} inside.",
    },
};

const equationMapping: EquationMapping = {};

describe("BlockquoteReference", () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("renders a red error span without an anchor for unknown references", () => {
        vi.spyOn(console, "error").mockImplementation(() => {});

        const { container } = render(
            <BlockquoteReference
                href="#bqref-def:missing"
                blockquoteMapping={blockquoteMapping}
                equationMapping={equationMapping}
                previewBlockquotes
            />,
        );

        const errorSpan = screen.getByText(
            "Error: reference def:missing not labeled",
        );
        expect(errorSpan.tagName).toBe("SPAN");
        expect(errorSpan).toHaveStyle({ color: "rgb(255, 0, 0)" });
        expect(container.querySelector("a")).toBeNull();
    });

    it("exposes a text alternative on the error span so meaning is not color-only", () => {
        vi.spyOn(console, "error").mockImplementation(() => {});

        render(
            <BlockquoteReference
                href="#bqref-def:missing"
                blockquoteMapping={blockquoteMapping}
                equationMapping={equationMapping}
            />,
        );

        const errorSpan = screen.getByText(
            "Error: reference def:missing not labeled",
        );
        expect(errorSpan).toHaveTextContent(/unresolved reference/i);
        expect(errorSpan.querySelector("span.sr-only")).not.toBeNull();
    });

    it("gives reference anchors an accessible name via getByRole", () => {
        render(
            <BlockquoteReference
                href="#bqref-def:first"
                blockquoteMapping={blockquoteMapping}
                equationMapping={equationMapping}
            />,
        );

        const link = screen.getByRole("link", { name: /definition 1\.1/i });
        expect(link).toHaveAttribute("href", "#def:first");
        expect(link).toHaveAttribute("aria-label", "Definition 1.1");
    });

    it("keeps the accessible name on the tooltip-wrapped anchor", () => {
        render(
            <BlockquoteReference
                href="#bqref-def:first"
                blockquoteMapping={blockquoteMapping}
                equationMapping={equationMapping}
                previewBlockquotes
            />,
        );

        expect(
            screen.getByRole("link", { name: /definition 1\.1/i }),
        ).toHaveAttribute("href", "#def:first");
    });

    it("renders a capitalized type label linking to the blockquote", () => {
        const { container } = render(
            <BlockquoteReference
                href="#bqref-def:first"
                blockquoteMapping={blockquoteMapping}
                equationMapping={equationMapping}
            />,
        );

        const anchor = container.querySelector('a[href="#def:first"]');
        expect(anchor).toHaveTextContent("Definition 1.1");
    });

    it("shows a preview card on hover when previewBlockquotes is true", async () => {
        render(
            <BlockquoteReference
                href="#bqref-def:first"
                blockquoteMapping={blockquoteMapping}
                equationMapping={equationMapping}
                previewBlockquotes
            />,
        );

        fireEvent.mouseEnter(screen.getByRole("link"));

        await waitFor(() => {
            expect(
                screen.getByText(/A reusable definition/),
            ).toBeInTheDocument();
        });
        // \label{...} is stripped from the preview content.
        expect(screen.queryByText(/\\label/)).toBeNull();
        expect(screen.getByText("Definition 1.1.")).toBeInTheDocument();
    });

    it("does not show a preview card when previewBlockquotes is false", async () => {
        render(
            <BlockquoteReference
                href="#bqref-def:first"
                blockquoteMapping={blockquoteMapping}
                equationMapping={equationMapping}
                previewBlockquotes={false}
            />,
        );

        const anchor = screen.getByRole("link");
        fireEvent.mouseEnter(anchor);

        await waitFor(() => {
            expect(anchor).toHaveTextContent("Definition 1.1");
        });
        expect(screen.queryByText(/A reusable definition/)).toBeNull();
    });

    it("renders nested #bqref- links inside preview cards as styled references without nested previews", async () => {
        const nestedMapping: BlockquoteMapping = {
            ...blockquoteMapping,
            "thm:second": {
                blockquoteNumber: "1.2",
                blockquoteType: "theorem",
                blockquoteContent:
                    "Follows from [the definition](#bqref-def:first).",
            },
        };

        const { baseElement } = render(
            <BlockquoteReference
                href="#bqref-thm:second"
                blockquoteMapping={nestedMapping}
                equationMapping={equationMapping}
                previewBlockquotes
            />,
        );

        fireEvent.mouseEnter(screen.getByRole("link", { name: "Theorem 1.2" }));

        // The nested reference resolves to a BlockquoteReference link with
        // the resolved label, not a plain "the definition" anchor.
        const nestedLink = await screen.findByRole("link", {
            name: "Definition 1.1",
        });
        expect(nestedLink).toHaveAttribute("href", "#def:first");
        expect(screen.queryByText("the definition")).toBeNull();

        // The nested reference must not spawn its own preview card on hover
        // (previewBlockquotes is disabled for nested references).
        fireEvent.mouseEnter(nestedLink);
        await waitFor(() => {
            expect(nestedLink).toHaveTextContent("Definition 1.1");
        });
        expect(baseElement.textContent?.includes("A reusable definition")).toBe(
            false,
        );
    });
});
