import { render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import TikZ from "./TikZ";

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((promiseResolve) => {
        resolve = promiseResolve;
    });

    return { promise, resolve };
}

describe("TikZ", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
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
});
