import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_CODE_RUNNER_URL, runCode } from "./api";

function jsonResponse(body: unknown, init?: ResponseInit): Response {
    return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
        ...init,
    });
}

describe("runCode", () => {
    const fetchMock = vi.fn<typeof fetch>();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal("fetch", fetchMock);
        vi.spyOn(console, "error").mockImplementation(() => {});
    });

    afterEach(() => {
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
        vi.useRealTimers();
    });

    it("uses the default endpoint when no baseUrl is provided", async () => {
        fetchMock.mockResolvedValue(jsonResponse({ output: "hi", image: "" }));

        const result = await runCode("print('hi')", "python");

        expect(fetchMock).toHaveBeenCalledTimes(1);
        expect(fetchMock).toHaveBeenCalledWith(
            `${DEFAULT_CODE_RUNNER_URL}/api/coderunner`,
            expect.objectContaining({
                method: "POST",
                body: JSON.stringify({
                    code: "print('hi')",
                    language: "python",
                }),
            }),
        );
        expect(result).toEqual({ output: "hi", image: "" });
    });

    it("uses a custom endpoint when baseUrl is provided", async () => {
        fetchMock.mockResolvedValue(jsonResponse({ output: "ok", image: "" }));

        await runCode("print('hi')", "python", {
            baseUrl: "https://runner.example.com",
        });

        expect(fetchMock).toHaveBeenCalledWith(
            "https://runner.example.com/api/coderunner",
            expect.objectContaining({ method: "POST" }),
        );
    });

    it("aborts the request when the timeout elapses", async () => {
        vi.useFakeTimers();
        fetchMock.mockImplementation(
            (_url, init) =>
                new Promise<Response>((_resolve, reject) => {
                    init?.signal?.addEventListener("abort", () => {
                        reject(
                            (init.signal as AbortSignal).reason ??
                                new Error("aborted"),
                        );
                    });
                }),
        );

        const promise = runCode("while True: pass", "python", {
            timeoutMs: 5000,
        });
        const assertion = expect(promise).rejects.toThrow(
            "Code execution timed out after 5000ms",
        );

        await vi.advanceTimersByTimeAsync(5001);
        await assertion;
    });

    it("aborts the request when an external signal is aborted", async () => {
        fetchMock.mockImplementation(
            (_url, init) =>
                new Promise<Response>((_resolve, reject) => {
                    init?.signal?.addEventListener("abort", () => {
                        reject(
                            (init.signal as AbortSignal).reason ??
                                new Error("aborted"),
                        );
                    });
                }),
        );

        const controller = new AbortController();
        const promise = runCode("print('hi')", "python", {
            signal: controller.signal,
        });
        const assertion = expect(promise).rejects.toThrow("unmounted");

        controller.abort(new Error("unmounted"));
        await assertion;
    });

    it("falls back to a status-based message for a non-JSON 502 response", async () => {
        fetchMock.mockResolvedValue(
            new Response("<html>Bad Gateway</html>", {
                status: 502,
                statusText: "Bad Gateway",
                headers: { "Content-Type": "text/html" },
            }),
        );

        await expect(runCode("print('hi')", "python")).rejects.toThrow(
            "HTTP 502 Bad Gateway",
        );
    });

    it("falls back to a status-based message when a JSON error body is malformed", async () => {
        fetchMock.mockResolvedValue(
            new Response("not-json", {
                status: 500,
                statusText: "Internal Server Error",
                headers: { "Content-Type": "application/json" },
            }),
        );

        await expect(runCode("print('hi')", "python")).rejects.toThrow(
            "HTTP 500 Internal Server Error",
        );
    });

    it("uses the server-provided error message when available", async () => {
        fetchMock.mockResolvedValue(
            jsonResponse(
                { error: "Unsupported language" },
                { status: 400, statusText: "Bad Request" },
            ),
        );

        await expect(runCode("print('hi')", "brainfuck")).rejects.toThrow(
            "Unsupported language",
        );
    });
});
