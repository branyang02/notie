import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EquationMapping } from "../utils/utils";
import EquationReference from "./EquationReference";

const equationMapping: EquationMapping = {
    "eq:first": {
        equationNumber: "1.1",
        equationString:
            "\\begin{equation} x = 1 \\label{eq:first} \\end{equation}",
    },
};

describe("EquationReference", () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("renders a red error span without an anchor for unknown references", () => {
        vi.spyOn(console, "error").mockImplementation(() => {});

        const { container } = render(
            <EquationReference
                href="#pre-eqn-eqref:eq:missing"
                equationMapping={equationMapping}
                previewEquation
            />,
        );

        const errorSpan = screen.getByText(
            "(Error: reference eq:missing not labeled)",
        );
        expect(errorSpan.tagName).toBe("SPAN");
        expect(errorSpan).toHaveStyle({ color: "rgb(255, 0, 0)" });
        expect(container.querySelector("a")).toBeNull();
    });

    it("renders parenthesized numbers for eqref references", () => {
        const { container } = render(
            <EquationReference
                href="#pre-eqn-eqref:eq:first"
                equationMapping={equationMapping}
            />,
        );

        const anchor = container.querySelector('a[href="#eqn-1.1"]');
        expect(anchor).toHaveTextContent("(1.1)");
    });

    it("renders bare numbers for ref references", () => {
        const { container } = render(
            <EquationReference
                href="#pre-eqn-ref:eq:first"
                equationMapping={equationMapping}
            />,
        );

        const anchor = container.querySelector('a[href="#eqn-1.1"]');
        expect(anchor?.textContent).toBe("1.1");
    });

    it("shows an equation preview tooltip on hover when previewEquation is true", async () => {
        render(
            <EquationReference
                href="#pre-eqn-eqref:eq:first"
                equationMapping={equationMapping}
                previewEquation
            />,
        );

        fireEvent.mouseEnter(screen.getByRole("link"));

        await waitFor(() => {
            expect(document.querySelector(".katex")).not.toBeNull();
        });
        // The preview strips \label before rendering, so KaTeX must not error.
        expect(document.querySelector(".katex-error")).toBeNull();
    });

    it("renders a plain anchor without a tooltip when previewEquation is false", async () => {
        render(
            <EquationReference
                href="#pre-eqn-eqref:eq:first"
                equationMapping={equationMapping}
                previewEquation={false}
            />,
        );

        const anchor = screen.getByRole("link");
        fireEvent.mouseEnter(anchor);

        await waitFor(() => {
            expect(anchor).toHaveTextContent("(1.1)");
        });
        expect(document.querySelector(".katex")).toBeNull();
    });
});
