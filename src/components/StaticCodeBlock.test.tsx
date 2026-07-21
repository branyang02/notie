import { render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const highlightWithCacheMock = vi.fn();

vi.mock("../utils/shikiHighlighter", async () => {
    const actual = await vi.importActual<
        typeof import("../utils/shikiHighlighter")
    >("../utils/shikiHighlighter");
    return {
        ...actual,
        highlightWithCache: (...args: unknown[]) =>
            highlightWithCacheMock(...args),
    };
});

import StaticCodeBlock from "./StaticCodeBlock";

describe("StaticCodeBlock", () => {
    afterEach(() => {
        vi.restoreAllMocks();
        highlightWithCacheMock.mockReset();
    });

    it("renders HTML-escaped code as the initial fallback", () => {
        // A highlighter that never resolves keeps the fallback visible.
        highlightWithCacheMock.mockReturnValue(new Promise(() => {}));

        const { container } = render(
            <StaticCodeBlock
                code={'<script>alert("x") && 1 > 0</script>'}
                language="html"
                theme="github-light"
            />,
        );

        const codeElement = container.querySelector("pre > code");
        expect(codeElement).not.toBeNull();
        expect(codeElement!.innerHTML).toBe(
            '&lt;script&gt;alert("x") &amp;&amp; 1 &gt; 0&lt;/script&gt;',
        );
        // The raw markup must not become live DOM nodes.
        expect(container.querySelector("script")).toBeNull();
    });

    it("keeps the escaped fallback when highlighting fails", async () => {
        highlightWithCacheMock.mockRejectedValue(
            new Error("unsupported language"),
        );

        const { container } = render(
            <StaticCodeBlock
                code="a < b"
                language="klingon"
                theme="github-light"
            />,
        );

        await waitFor(() => {
            expect(highlightWithCacheMock).toHaveBeenCalled();
        });

        const codeElement = container.querySelector("pre > code");
        expect(codeElement!.innerHTML).toBe("a &lt; b");
    });

    it("swaps in highlighter output once highlighting resolves", async () => {
        highlightWithCacheMock.mockResolvedValue(
            '<pre class="shiki"><code>print(1)</code></pre>',
        );

        const { container } = render(
            <StaticCodeBlock
                code="print(1)"
                language="py"
                theme="github-dark"
            />,
        );

        await waitFor(() => {
            expect(container.querySelector("pre.shiki")).not.toBeNull();
        });
        expect(highlightWithCacheMock).toHaveBeenCalledWith(
            "print(1)",
            "py",
            "github-dark",
        );
    });
});
