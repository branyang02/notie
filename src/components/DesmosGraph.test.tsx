import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// DesmosGraph caches its script-loader promise at module scope, so import a
// fresh copy for each test to avoid state leaking between tests.
async function importDesmosGraph() {
    vi.resetModules();
    const module = await import("./DesmosGraph");
    return module.default;
}

function stubDesmos() {
    const destroy = vi.fn();
    const setExpression = vi.fn();
    const GraphingCalculator = vi.fn(() => ({ destroy, setExpression }));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).Desmos = { GraphingCalculator };

    return { GraphingCalculator, destroy, setExpression };
}

describe("DesmosGraph", () => {
    beforeEach(() => {
        vi.spyOn(console, "error").mockImplementation(() => {});
    });

    afterEach(() => {
        vi.restoreAllMocks();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (window as any).Desmos;
        document.head.innerHTML = "";
        document.body.innerHTML = "";
    });

    it("creates the calculator without inverted colors by default", async () => {
        const { GraphingCalculator } = stubDesmos();
        const DesmosGraph = await importDesmosGraph();

        render(<DesmosGraph graphScript="y=x" />);

        await waitFor(() => {
            expect(GraphingCalculator).toHaveBeenCalledTimes(1);
        });
        expect(GraphingCalculator).toHaveBeenCalledWith(
            expect.any(HTMLElement),
            { invertedColors: false },
        );
    });

    it("creates the calculator with inverted colors in dark mode", async () => {
        const { GraphingCalculator } = stubDesmos();
        const DesmosGraph = await importDesmosGraph();

        render(<DesmosGraph graphScript="y=x" appearance="dark" />);

        await waitFor(() => {
            expect(GraphingCalculator).toHaveBeenCalledTimes(1);
        });
        expect(GraphingCalculator).toHaveBeenCalledWith(
            expect.any(HTMLElement),
            { invertedColors: true },
        );
    });

    it("sets one expression per non-empty line", async () => {
        const { setExpression } = stubDesmos();
        const DesmosGraph = await importDesmosGraph();

        render(<DesmosGraph graphScript={"y=x\n\ny=x^2"} />);

        await waitFor(() => {
            expect(setExpression).toHaveBeenCalledTimes(2);
        });
        expect(setExpression).toHaveBeenNthCalledWith(1, {
            id: "graph0",
            latex: "y=x",
        });
        expect(setExpression).toHaveBeenNthCalledWith(2, {
            id: "graph1",
            latex: "y=x^2",
        });
    });

    it("destroys the calculator on unmount", async () => {
        const { GraphingCalculator, destroy } = stubDesmos();
        const DesmosGraph = await importDesmosGraph();

        const { unmount } = render(<DesmosGraph graphScript="y=x" />);

        await waitFor(() => {
            expect(GraphingCalculator).toHaveBeenCalledTimes(1);
        });

        unmount();
        expect(destroy).toHaveBeenCalledTimes(1);
    });

    it("renders a fallback when the Desmos script fails to load", async () => {
        const DesmosGraph = await importDesmosGraph();

        // No window.Desmos stub: the component injects a <script> tag and
        // waits for onload/onerror. Simulate a network failure.
        const appendChild = vi
            .spyOn(document.head, "appendChild")
            .mockImplementation(((node: Node) => {
                if (node instanceof HTMLScriptElement) {
                    setTimeout(() => {
                        node.onerror?.(new Event("error"));
                    }, 0);
                }
                return node;
            }) as typeof document.head.appendChild);

        render(<DesmosGraph graphScript="y=x" />);

        const fallback = await screen.findByTestId("desmos-fallback");
        expect(fallback).toHaveTextContent("Failed to load Desmos");
        expect(console.error).toHaveBeenCalled();

        appendChild.mockRestore();
    });
});
