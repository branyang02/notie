import {
    fireEvent,
    render,
    screen,
    waitFor,
    within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Notie from "./Notie";

describe("Notie", () => {
    beforeEach(() => {
        window.history.replaceState(null, "", "/");
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("scrolls to a debounce-labeled equation anchor targeted by the URL hash", async () => {
        // Equation ids (eqn-X.Y) are assigned by a coalesced DOM-labeling
        // effect that trails the last progressive reveal by a short
        // debounce, so the pending-scroll effect must retry until the
        // anchor exists instead of giving up on its first run.
        window.history.replaceState(null, "", "/#eqn-3.1");
        const scrollSpy = vi.spyOn(HTMLElement.prototype, "scrollIntoView");

        render(
            <Notie
                markdown={`# Demo

## One

First section.

## Two

Second section.

## Three

$$
\\begin{equation} \\label{eq:last}
z = 3
\\end{equation}
$$
`}
            />,
        );

        // The target lives in a not-yet-revealed section, and its id only
        // appears after the post-reveal labeling debounce.
        expect(document.getElementById("eqn-3.1")).toBeNull();

        await waitFor(
            () => {
                expect(scrollSpy).toHaveBeenCalled();
            },
            { timeout: 3000 },
        );

        const scrolledElement = scrollSpy.mock
            .contexts[0] as unknown as HTMLElement;
        expect(scrolledElement.id).toBe("eqn-3.1");
        expect(window.location.hash).toBe("#eqn-3.1");
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

    it("renders single-line equation environments as display math with a working eqref anchor", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

## First Section

$$\\begin{equation}\\label{eq:x} y = 1 \\end{equation}$$

See $\\eqref{eq:x}$.
`}
            />,
        );

        // remark-math must parse the normalized equation as display math:
        // a katex-display block and no red ParseError ("{equation} can be
        // used only in display mode").
        expect(container.querySelector(".katex-display")).toBeInTheDocument();
        expect(container.querySelector(".katex-error")).not.toBeInTheDocument();

        // The eqref link resolves to a real equation anchor in the DOM,
        // not a dangling href.
        const equationReference = container.querySelector('a[href="#eqn-1.1"]');
        expect(equationReference).toHaveTextContent("(1.1)");
        expect(container.querySelector("#eqn-1\\.1")).toHaveTextContent(
            "(1.1)",
        );
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

    it("resolves references to proof and note blockquotes with numbers", () => {
        render(
            <Notie
                markdown={`# Demo

## First Section

<blockquote class="proof" id="proof:main">
Proof body.
</blockquote>

<blockquote class="note" id="note:main">
Note body.
</blockquote>

See [proof](#bqref-proof:main) and [note](#bqref-note:main).
`}
            />,
        );

        const proofReference = screen
            .getAllByRole("link")
            .find((link) => link.getAttribute("href") === "#proof:main");
        expect(proofReference).toHaveTextContent("Proof 1.1");

        const noteReference = screen
            .getAllByRole("link")
            .find((link) => link.getAttribute("href") === "#note:main");
        expect(noteReference).toHaveTextContent("Note 1.1");
    });

    it("sets dark blockquote CSS variables and themes preview cards in dark mode", async () => {
        const { baseElement } = render(
            <Notie
                markdown={`# Demo

## First Section

<blockquote class="definition" id="def:dark">
Dark definition.
</blockquote>

See [definition](#bqref-def:dark).
`}
                theme="default dark"
            />,
        );

        // Dark appearance swaps the blockquote variables on the root element.
        const rootStyle = document.documentElement.style;
        expect(rootStyle.getPropertyValue("--blog-bq-definition-bg")).toBe(
            "rgba(120, 220, 80, 0.14)",
        );
        expect(rootStyle.getPropertyValue("--blog-bq-definition-label")).toBe(
            "#6ee76a",
        );
        expect(rootStyle.getPropertyValue("--blog-bq-shadow")).toContain(
            "rgba(0, 0, 0, 0.5)",
        );

        // The preview card consumes the same variables and inherits the
        // theme text color, so its content stays readable in dark mode.
        const definitionReference = screen.getByRole("link", {
            name: "Definition 1.1",
        });
        fireEvent.mouseEnter(definitionReference);

        await waitFor(() => {
            expect(
                within(baseElement).getByText("Definition 1.1."),
            ).toBeInTheDocument();
        });

        const card = within(baseElement)
            .getByText("Definition 1.1.")
            .closest("div") as HTMLElement;
        expect(card.style.backgroundImage).toContain(
            "var(--blog-bq-definition-bg",
        );
        expect(card.style.color).toBe("var(--blog-text-color)");
        const cardLabel = within(baseElement).getByText("Definition 1.1.");
        expect(cardLabel.style.color).toContain(
            "var(--blog-bq-definition-label",
        );
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

    it("gives every TOC entry an id that exists in the rendered DOM (tricky headings)", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

## Tricky Headings

### C & D

### It's \`code\` (v2)

### Setup

### Setup

### See [the docs](https://example.com) here

### **Bold** heading
`}
            />,
        );

        const toc = screen.getByRole("navigation", { name: /contents/i });
        const ids = Array.from(toc.querySelectorAll("a[href^='#']")).map(
            (link) => link.getAttribute("href")!.slice(1),
        );

        expect(ids).toEqual([
            "tricky-headings",
            "c--d",
            "its-code-v2",
            "setup",
            "setup-1",
            "see-the-docs-here",
            "bold-heading",
        ]);

        const content = container.querySelector(
            "[class*='blog-content']",
        ) as HTMLElement;
        for (const id of ids) {
            expect(
                content.querySelector(`[id="${CSS.escape(id)}"]`),
                `expected a rendered element with id "${id}"`,
            ).not.toBeNull();
        }
    });

    it("gives every TOC entry an id that exists in the DOM for reference-style-link headings", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

## Reference Links

### See [the guide][guide] here

### Read [the spec][] now

### Try [shortcut] form

### Try [shortcut] form

[guide]: https://example.com/guide
[the spec]: https://example.com/spec
[shortcut]: https://example.com/shortcut
`}
            />,
        );

        const toc = screen.getByRole("navigation", { name: /contents/i });
        const ids = Array.from(toc.querySelectorAll("a[href^='#']")).map(
            (link) => link.getAttribute("href")!.slice(1),
        );

        expect(ids).toEqual([
            "reference-links",
            "see-the-guide-here",
            "read-the-spec-now",
            "try-shortcut-form",
            "try-shortcut-form-1",
        ]);

        const content = container.querySelector(
            "[class*='blog-content']",
        ) as HTMLElement;
        for (const id of ids) {
            expect(
                content.querySelector(`[id="${CSS.escape(id)}"]`),
                `expected a rendered element with id "${id}"`,
            ).not.toBeNull();
        }
    });

    it("gives every TOC entry an id that exists in the DOM with numbered headings", async () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

## Setup

### Install

## Setup

### Install

## C & D
`}
                config={{ theme: { numberedHeading: true } }}
            />,
        );

        const toc = screen.getByRole("navigation", { name: /contents/i });
        const ids = Array.from(toc.querySelectorAll("a[href^='#']")).map(
            (link) => link.getAttribute("href")!.slice(1),
        );
        expect(ids).toEqual([
            "1setup",
            "11install",
            "2setup",
            "21install",
            "3c--d",
        ]);

        // Force all sections to render (Notie renders large notes
        // progressively) by navigating to the last TOC entry.
        const lastLink = within(toc)
            .getAllByRole("link")
            .find(
                (link) =>
                    link.getAttribute("href") === `#${ids[ids.length - 1]}`,
            )!;
        fireEvent.click(lastLink);

        const content = container.querySelector(
            "[class*='blog-content']",
        ) as HTMLElement;
        await waitFor(() => {
            for (const id of ids) {
                expect(
                    content.querySelector(`[id="${CSS.escape(id)}"]`),
                    `expected a rendered element with id "${id}"`,
                ).not.toBeNull();
            }
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

    it("strips javascript: URLs from markdown links", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

[click me](javascript:alert(1))

[obfuscated](JaVaScRiPt:alert(1))

[safe](https://example.com)

[anchor](#some-anchor)
`}
            />,
        );

        for (const anchor of Array.from(container.querySelectorAll("a"))) {
            const href = anchor.getAttribute("href") ?? "";
            expect(href.toLowerCase()).not.toContain("javascript:");
        }

        const safeLink = screen.getByRole("link", { name: "safe" });
        expect(safeLink).toHaveAttribute("href", "https://example.com");
        const anchorLink = screen.getByRole("link", { name: "anchor" });
        expect(anchorLink).toHaveAttribute("href", "#some-anchor");
    });

    it("does not render javascript: hrefs from KaTeX \\href commands", () => {
        const { container } = render(
            <Notie
                markdown={`# Demo

$\\href{javascript:alert(1)}{evil}$

$\\href{https://example.com}{good}$
`}
            />,
        );

        for (const anchor of Array.from(container.querySelectorAll("a"))) {
            const href = anchor.getAttribute("href") ?? "";
            expect(href.toLowerCase()).not.toContain("javascript:");
        }

        expect(
            container.querySelector('a[href="https://example.com"]'),
        ).not.toBeNull();
    });

    it("does not render the TOC for empty markdown", () => {
        render(<Notie markdown="" />);

        expect(screen.queryByRole("navigation")).toBeNull();
    });

    it("does not render the TOC for title-only documents", () => {
        render(<Notie markdown={"# Only a Title\n\nSome body text.\n"} />);

        expect(screen.getByText("Some body text.")).toBeInTheDocument();
        expect(screen.queryByRole("navigation")).toBeNull();
    });
});
