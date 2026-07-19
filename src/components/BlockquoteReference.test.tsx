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
});
