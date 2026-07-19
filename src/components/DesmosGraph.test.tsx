import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// DesmosGraph caches its script-loader promises at module scope, so import a
// fresh copy for each test to avoid state leaking between tests.
async function importDesmosGraph() {
    vi.resetModules();
    const module = await import("./DesmosGraph");
    return module;
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
        const { default: DesmosGraph } = await importDesmosGraph();

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
        const { default: DesmosGraph } = await importDesmosGraph();

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
        const { default: DesmosGraph } = await importDesmosGraph();

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
        const { default: DesmosGraph } = await importDesmosGraph();

        const { unmount } = render(<DesmosGraph graphScript="y=x" />);

        await waitFor(() => {
            expect(GraphingCalculator).toHaveBeenCalledTimes(1);
        });

        unmount();
        expect(destroy).toHaveBeenCalledTimes(1);
    });

    // Intercept injected <script> tags: record their src and fire onload
    // (defining window.Desmos just before, like the real script would).
    function interceptScriptInjection() {
        const scriptSrcs: string[] = [];
        const appendChild = vi
            .spyOn(document.head, "appendChild")
            .mockImplementation(((node: Node) => {
                if (node instanceof HTMLScriptElement) {
                    scriptSrcs.push(node.src);
                    setTimeout(() => {
                        stubDesmos();
                        node.onload?.(new Event("load"));
                    }, 0);
                }
                return node;
            }) as typeof document.head.appendChild);
        return { scriptSrcs, appendChild };
    }

    it("loads the Desmos script with the default demo API key", async () => {
        const { default: DesmosGraph, DEFAULT_DESMOS_API_KEY } =
            await importDesmosGraph();
        const { scriptSrcs, appendChild } = interceptScriptInjection();

        render(<DesmosGraph graphScript="y=x" />);

        await waitFor(() => {
            expect(scriptSrcs).toHaveLength(1);
        });
        expect(scriptSrcs[0]).toBe(
            `https://www.desmos.com/api/v1.9/calculator.js?apiKey=${DEFAULT_DESMOS_API_KEY}`,
        );

        appendChild.mockRestore();
    });

    it("loads the Desmos script with a custom API key", async () => {
        const { default: DesmosGraph } = await importDesmosGraph();
        const { scriptSrcs, appendChild } = interceptScriptInjection();

        render(<DesmosGraph graphScript="y=x" apiKey="my-custom-key" />);

        await waitFor(() => {
            expect(scriptSrcs).toHaveLength(1);
        });
        expect(scriptSrcs[0]).toBe(
            "https://www.desmos.com/api/v1.9/calculator.js?apiKey=my-custom-key",
        );

        appendChild.mockRestore();
    });

    it("caches the script loader per API key", async () => {
        const { default: DesmosGraph } = await importDesmosGraph();

        // Record injected scripts but never fire onload, so the loader
        // promises stay pending and caching behavior is observable.
        const scriptSrcs: string[] = [];
        const appendChild = vi
            .spyOn(document.head, "appendChild")
            .mockImplementation(((node: Node) => {
                if (node instanceof HTMLScriptElement) {
                    scriptSrcs.push(node.src);
                }
                return node;
            }) as typeof document.head.appendChild);

        render(
            <>
                <DesmosGraph graphScript="y=x" apiKey="key-a" />
                <DesmosGraph graphScript="y=x^2" apiKey="key-a" />
                <DesmosGraph graphScript="y=x^3" apiKey="key-b" />
            </>,
        );

        await waitFor(() => {
            expect(scriptSrcs).toHaveLength(2);
        });
        expect(scriptSrcs).toEqual([
            "https://www.desmos.com/api/v1.9/calculator.js?apiKey=key-a",
            "https://www.desmos.com/api/v1.9/calculator.js?apiKey=key-b",
        ]);

        appendChild.mockRestore();
    });

    it("renders a fallback when the Desmos script fails to load", async () => {
        const { default: DesmosGraph } = await importDesmosGraph();

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
