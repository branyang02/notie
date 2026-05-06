import {
    fireEvent,
    render,
    screen,
    waitFor,
    within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import Notie from "./Notie";

describe("Notie", () => {
    beforeEach(() => {
        window.history.replaceState(null, "", "/");
    });

    it("renders markdown, TOC, equations, blockquote references, and custom components", async () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

## First Section

$$
\\begin{equation} \\label{eq:first}
x = 1
\\end{equation}
$$

See $\\eqref{eq:first}$.

<blockquote class="definition" id="def:first">
Reusable definition.
</blockquote>

See [definition](#bqref-def:first).

\`\`\`component
{
    componentName: "Widget"
}
\`\`\`
`}
                config={{
                    theme: {
                        blockquoteStyle: "latex",
                        numberedHeading: true,
                    },
                }}
                customComponents={{
                    Widget: () => <div data-testid="custom-widget">Widget</div>,
                }}
            />,
        );

        expect(
            screen.getByRole("heading", { name: /Demo/i }),
        ).toBeInTheDocument();
        expect(screen.getByTestId("custom-widget")).toHaveTextContent("Widget");
        expect(screen.getByText("Definition 1.1")).toBeInTheDocument();

        const toc = screen.getByRole("navigation", { name: /contents/i });
        expect(within(toc).getByText(/First Section/)).toBeInTheDocument();

        const equationReference = container.querySelector('a[href="#eqn-1.1"]');
        expect(equationReference).toHaveTextContent("(1.1)");
        expect(container.querySelector("#eqn-1\\.1")).toHaveTextContent(
            "(1.1)",
        );

        const blockquoteReference = screen
            .getAllByRole("link")
            .find((link) => link.getAttribute("href") === "#def:first");
        expect(blockquoteReference).toHaveTextContent("Definition 1.1");
    });

    it("renders equation references whose labels contain underscores", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

## First Section

$$
\\begin{equation} \\label{eq:k_step_q_estimate}
\\hat{Q}^{(k)}(s_t, a_t) = r_t + \\gamma V_\\phi(s_{t+1})
\\end{equation}
$$

See $\\eqref{eq:k_step_q_estimate}$ and $\\ref{eq:k_step_q_estimate}$.
`}
            />,
        );

        const references = Array.from(
            container.querySelectorAll('a[href="#eqn-1.1"]'),
        );
        expect(references.map((ref) => ref.textContent)).toEqual([
            "(1.1)",
            "1.1",
        ]);
        expect(container.querySelector("#eqn-1\\.1")).toHaveTextContent(
            "(1.1)",
        );
        expect(container.querySelector(".katex-error")).not.toBeInTheDocument();
    });

    it("resolves equation references inside blockquote preview cards", async () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

## First Section

$$
\\begin{equation} \\label{eq:k_step_q_estimate}
\\hat{Q}^{(k)}(s_t, a_t) = r_t + \\gamma V_\\phi(s_{t+1})
\\end{equation}
$$

<blockquote class="algorithm" id="alg:critic-target">

Use $\\eqref{eq:k_step_q_estimate}$ as the critic target.

</blockquote>

See [Algorithm](#bqref-alg:critic-target).
`}
                config={{
                    theme: {
                        blockquoteStyle: "latex",
                    },
                }}
            />,
        );

        const algorithmReference = screen.getByRole("link", {
            name: "Algorithm 1.1",
        });
        fireEvent.mouseEnter(algorithmReference);

        await waitFor(() => {
            const equationReferences = Array.from(
                container.querySelectorAll('a[href="#eqn-1.1"]'),
            );
            expect(
                equationReferences.some((link) => link.textContent === "(1.1)"),
            ).toBe(true);
        });
        expect(container.querySelector(".katex-error")).not.toBeInTheDocument();
    });

    it("renders large notes progressively while preserving TOC navigation", async () => {
        render(
            <Notie
                markdown={`# Demo

## One

First section.

## Two

Second section.

## Three

Third section.
`}
            />,
        );

        expect(screen.getByText("First section.")).toBeInTheDocument();
        expect(screen.queryByText("Third section.")).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole("link", { name: "Three" }));

        await waitFor(() => {
            expect(screen.getByText("Third section.")).toBeInTheDocument();
        });
    });

    it("rerenders when markdown content changes", () => {
        const { rerender } = render(
            <Notie markdown={"# Original\n\n## Section\n\nOriginal body."} />,
        );

        expect(screen.getByText("Original body.")).toBeInTheDocument();

        rerender(
            <Notie markdown={"# Updated\n\n## Section\n\nUpdated body."} />,
        );

        expect(screen.getByText("Updated body.")).toBeInTheDocument();
        expect(screen.queryByText("Original body.")).not.toBeInTheDocument();
    });

    it("keeps already rendered lower sections mounted across markdown updates", async () => {
        const { rerender } = render(
            <Notie
                markdown={`# Demo

## One

First section.

## Two

Second section.

## Three

Original third section.
`}
            />,
        );

        await waitFor(() => {
            expect(
                screen.getByText("Original third section."),
            ).toBeInTheDocument();
        });

        rerender(
            <Notie
                markdown={`# Demo

## One

First section.

## Two

Second section.

## Three

Updated third section.
`}
            />,
        );

        expect(screen.getByText("Updated third section.")).toBeInTheDocument();
        expect(
            screen.queryByText("Original third section."),
        ).not.toBeInTheDocument();
    });

    it("resets progressive rendering when markdown changes to a different document", async () => {
        const { rerender } = render(
            <Notie
                markdown={`# Demo

## One

First section.

## Two

Second section.

## Three

Original third section.
`}
            />,
        );

        await waitFor(() => {
            expect(
                screen.getByText("Original third section."),
            ).toBeInTheDocument();
        });

        rerender(
            <Notie
                markdown={`# Different Demo

## Alpha

Alpha section.

## Beta

Beta section.

## Gamma

New third section.
`}
            />,
        );

        expect(screen.getByText("Alpha section.")).toBeInTheDocument();
        expect(
            screen.queryByText("New third section."),
        ).not.toBeInTheDocument();
    });
});
