import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const PINNED_SHA = "1d1f0844bd918e09e7eac081f86a70ba28635301";

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((promiseResolve) => {
        resolve = promiseResolve;
    });

    return { promise, resolve };
}

// TikZ caches loader promises at module scope, so import a fresh copy for
// each test to avoid state leaking between tests.
async function importTikZ() {
    vi.resetModules();
    const module = await import("./TikZ");
    return module.default;
}

describe("TikZ", () => {
    beforeEach(() => {
        vi.spyOn(console, "error").mockImplementation(() => {});
    });

    afterEach(() => {
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
        document.head.innerHTML = "";
        document.body.innerHTML = "";
    });

    it("waits for the TikZJax loader before appending text/tikz scripts", async () => {
        const scriptText = deferred<string>();
        const fetchMock = vi.fn((url: string) => {
            if (url.includes("styles.css")) {
                return Promise.resolve({
                    ok: true,
                    text: () => Promise.resolve(".tikz-drawing {}"),
                });
            }

            return Promise.resolve({
                ok: true,
                text: () => scriptText.promise,
            });
        });

        vi.stubGlobal("fetch", fetchMock);

        const TikZ = await importTikZ();
        const { container } = render(
            <TikZ tikzScript="\\begin{tikzpicture}\\end{tikzpicture}" />,
        );

        expect(container.querySelector('script[type="text/tikz"]')).toBeNull();

        scriptText.resolve("window.TikzJax = true;");

        await waitFor(() => {
            expect(
                container.querySelector('script[type="text/tikz"]'),
            ).toHaveTextContent("tikzpicture");
        });
    });

    it("fetches TikZJax assets pinned to a commit SHA instead of main", async () => {
        const requestedUrls: string[] = [];
        const fetchMock = vi.fn((url: string) => {
            requestedUrls.push(url);
            return Promise.resolve({
                ok: true,
                text: () => Promise.resolve(""),
            });
        });

        vi.stubGlobal("fetch", fetchMock);

        const TikZ = await importTikZ();
        render(<TikZ tikzScript="\\begin{tikzpicture}\\end{tikzpicture}" />);

        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledTimes(2);
        });

        const urls = requestedUrls;
        for (const url of urls) {
            expect(url).toContain(`/${PINNED_SHA}/`);
            expect(url).not.toContain("/main/");
        }
    });

    it("renders a fallback with the raw source when loading fails", async () => {
        const fetchMock = vi.fn(() =>
            Promise.reject(new Error("network down")),
        );
        vi.stubGlobal("fetch", fetchMock);

        const TikZ = await importTikZ();
        const source = "\\begin{tikzpicture}\\draw (0,0);\\end{tikzpicture}";
        render(<TikZ tikzScript={source} />);

        const fallback = await screen.findByTestId("tikz-fallback");
        expect(fallback).toHaveTextContent("TikZ rendering failed");
        expect(fallback.querySelector("pre")).toHaveTextContent(
            "\\draw (0,0);",
        );
        expect(console.error).toHaveBeenCalled();
    });

    it("retries loading on a new mount after a failure", async () => {
        const failingFetch = vi.fn(() =>
            Promise.reject(new Error("network down")),
        );
        vi.stubGlobal("fetch", failingFetch);

        const TikZ = await importTikZ();
        const { unmount } = render(
            <TikZ tikzScript="\\begin{tikzpicture}\\end{tikzpicture}" />,
        );

        await screen.findByTestId("tikz-fallback");
        expect(failingFetch).toHaveBeenCalledTimes(2);
        unmount();

        // The rejected promises must not be cached: a later mount should
        // fetch again and succeed.
        const succeedingFetch = vi.fn(() =>
            Promise.resolve({
                ok: true,
                text: () => Promise.resolve(""),
            }),
        );
        vi.stubGlobal("fetch", succeedingFetch);

        const { container } = render(
            <TikZ tikzScript="\\begin{tikzpicture}\\end{tikzpicture}" />,
        );

        await waitFor(() => {
            expect(succeedingFetch).toHaveBeenCalledTimes(2);
        });
        await waitFor(() => {
            expect(
                container.querySelector('script[type="text/tikz"]'),
            ).not.toBeNull();
        });
        expect(screen.queryByTestId("tikz-fallback")).toBeNull();
    });

    it("replaces the previous diagram when the source changes", async () => {
        const fetchMock = vi.fn(() =>
            Promise.resolve({
                ok: true,
                text: () => Promise.resolve(""),
            }),
        );
        vi.stubGlobal("fetch", fetchMock);

        const TikZ = await importTikZ();
        const { container, rerender } = render(
            <TikZ tikzScript="\\begin{tikzpicture}A\\end{tikzpicture}" />,
        );

        await waitFor(() => {
            expect(
                container.querySelector('script[type="text/tikz"]'),
            ).not.toBeNull();
        });

        // Simulate TikZJax replacing the script with a rendered SVG.
        const tikzContainer = container.querySelector(
            "div.tikz-drawing, [class*='tikz-drawing']",
        ) as HTMLDivElement;
        const staleSvg = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "svg",
        );
        tikzContainer.appendChild(staleSvg);

        rerender(<TikZ tikzScript="\\begin{tikzpicture}B\\end{tikzpicture}" />);

        await waitFor(() => {
            const scripts = tikzContainer.querySelectorAll(
                'script[type="text/tikz"]',
            );
            expect(scripts).toHaveLength(1);
            expect(scripts[0]).toHaveTextContent("B");
        });
        expect(tikzContainer.querySelector("svg")).toBeNull();
    });
});
