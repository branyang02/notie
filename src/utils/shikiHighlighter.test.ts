import { beforeEach, describe, expect, it, vi } from "vitest";

const codeToHtml = vi.fn(
    (code: string, options: { lang: string; theme: string }) =>
        `<pre data-theme="${options.theme}" data-lang="${options.lang}">${code}</pre>`,
);

vi.mock("shiki/core", () => ({
    createHighlighterCore: vi.fn(async () => ({ codeToHtml })),
}));

vi.mock("shiki/engine/javascript", () => ({
    createJavaScriptRegexEngine: vi.fn(() => ({})),
}));

import {
    __configureHighlightCacheForTests,
    highlightWithCache,
} from "./shikiHighlighter";

describe("highlightWithCache", () => {
    beforeEach(() => {
        __configureHighlightCacheForTests();
        codeToHtml.mockClear();
    });

    it("only calls the underlying highlighter once for identical inputs", async () => {
        const first = await highlightWithCache(
            "print('hi')",
            "python",
            "github-dark",
        );
        const second = await highlightWithCache(
            "print('hi')",
            "python",
            "github-dark",
        );

        expect(first).toBe(second);
        expect(codeToHtml).toHaveBeenCalledTimes(1);
    });

    it("re-highlights when the theme differs", async () => {
        await highlightWithCache("print('hi')", "python", "github-dark");
        await highlightWithCache("print('hi')", "python", "github-light");

        expect(codeToHtml).toHaveBeenCalledTimes(2);
    });

    it("re-highlights when the language differs", async () => {
        await highlightWithCache("x", "python", "github-dark");
        await highlightWithCache("x", "javascript", "github-dark");

        expect(codeToHtml).toHaveBeenCalledTimes(2);
    });

    it("re-highlights when the code differs", async () => {
        await highlightWithCache("a", "python", "github-dark");
        await highlightWithCache("b", "python", "github-dark");

        expect(codeToHtml).toHaveBeenCalledTimes(2);
    });

    it("evicts the oldest entry when the cap is exceeded", async () => {
        __configureHighlightCacheForTests(2);

        await highlightWithCache("one", "python", "github-dark");
        await highlightWithCache("two", "python", "github-dark");
        await highlightWithCache("three", "python", "github-dark");
        expect(codeToHtml).toHaveBeenCalledTimes(3);

        // "two" and "three" are still cached...
        await highlightWithCache("two", "python", "github-dark");
        await highlightWithCache("three", "python", "github-dark");
        expect(codeToHtml).toHaveBeenCalledTimes(3);

        // ...but "one" was evicted and must be re-highlighted.
        await highlightWithCache("one", "python", "github-dark");
        expect(codeToHtml).toHaveBeenCalledTimes(4);
    });

    it("refreshes recency on cache hits (LRU-ish eviction)", async () => {
        __configureHighlightCacheForTests(2);

        await highlightWithCache("one", "python", "github-dark");
        await highlightWithCache("two", "python", "github-dark");
        // Touch "one" so "two" becomes the oldest entry.
        await highlightWithCache("one", "python", "github-dark");
        await highlightWithCache("three", "python", "github-dark");
        expect(codeToHtml).toHaveBeenCalledTimes(3);

        // "one" survived the eviction; "two" did not.
        await highlightWithCache("one", "python", "github-dark");
        expect(codeToHtml).toHaveBeenCalledTimes(3);
        await highlightWithCache("two", "python", "github-dark");
        expect(codeToHtml).toHaveBeenCalledTimes(4);
    });

    it("does not cache more entries than the cap", async () => {
        __configureHighlightCacheForTests(3);

        for (let i = 0; i < 10; i++) {
            await highlightWithCache(`snippet-${i}`, "python", "github-dark");
        }
        expect(codeToHtml).toHaveBeenCalledTimes(10);

        // Only the 3 most recent entries are cached.
        await highlightWithCache("snippet-9", "python", "github-dark");
        await highlightWithCache("snippet-8", "python", "github-dark");
        await highlightWithCache("snippet-7", "python", "github-dark");
        expect(codeToHtml).toHaveBeenCalledTimes(10);

        await highlightWithCache("snippet-6", "python", "github-dark");
        expect(codeToHtml).toHaveBeenCalledTimes(11);
    });
});
