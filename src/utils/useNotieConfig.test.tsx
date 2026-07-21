import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { NotieConfig, NotieThemes } from "../config/NotieConfig";
import { useNotieConfig } from "./useNotieConfig";

describe("useNotieConfig", () => {
    it("returns a referentially stable config across re-renders with structurally equal inline configs", () => {
        const makeConfig = (): NotieConfig => ({
            fontSize: "1.2rem",
            theme: {
                linkColor: "#123456",
                numberedHeading: true,
            },
        });

        const { result, rerender } = renderHook(
            ({ config }: { config: NotieConfig }) =>
                useNotieConfig(config, "default"),
            { initialProps: { config: makeConfig() } },
        );

        const firstResult = result.current;

        // New object identity, same contents.
        rerender({ config: makeConfig() });
        expect(result.current).toBe(firstResult);

        rerender({ config: makeConfig() });
        expect(result.current).toBe(firstResult);

        // Changed contents must produce a new merged config.
        rerender({
            config: { ...makeConfig(), fontSize: "2rem" },
        });
        expect(result.current).not.toBe(firstResult);
        expect(result.current.fontSize).toBe("2rem");
    });

    it("keeps identity stable when no user config is provided", () => {
        const { result, rerender } = renderHook(() =>
            useNotieConfig(undefined, "default"),
        );

        const firstResult = result.current;
        rerender();
        expect(result.current).toBe(firstResult);
    });

    it("deep-merges a custom theme value into a predefined theme", () => {
        const { result } = renderHook(() =>
            useNotieConfig(
                { theme: { linkColor: "#ff0000" } },
                "Starlit Eclipse",
            ),
        );

        // The overridden value is applied...
        expect(result.current.theme.linkColor).toBe("#ff0000");
        // ...while the rest of the Starlit Eclipse theme is preserved.
        expect(result.current.theme.appearance).toBe("dark");
        expect(result.current.theme.backgroundColor).toBe("rgb(3 7 18)");
        expect(result.current.theme.fontFamily).toBe(
            "'Space Grotesk', sans-serif",
        );
        expect(result.current.theme.linkHoverColor).toBe("#f472b6");
        expect(result.current.theme.staticCodeTheme).toBe("github-dark");
    });

    describe("custom font links", () => {
        const getFontLinks = (href: string) =>
            Array.from(
                document.head.querySelectorAll<HTMLLinkElement>(
                    `link[rel="stylesheet"][href="${href}"]`,
                ),
            );

        it("appends both the content and TOC font links and removes them on unmount", () => {
            const fontUrl = "https://fonts.example.com/content-font.css";
            const tocFontUrl = "https://fonts.example.com/toc-font.css";
            const { unmount } = renderHook(() =>
                useNotieConfig({
                    theme: {
                        customFontUrl: fontUrl,
                        tocCustomFontUrl: tocFontUrl,
                    },
                }),
            );

            expect(getFontLinks(fontUrl)).toHaveLength(1);
            expect(getFontLinks(tocFontUrl)).toHaveLength(1);

            unmount();

            expect(getFontLinks(fontUrl)).toHaveLength(0);
            expect(getFontLinks(tocFontUrl)).toHaveLength(0);
        });

        it("loads both font links for a predefined theme that sets both URLs", () => {
            const themeName: NotieThemes = "Starlit Eclipse";
            const { result, unmount } = renderHook(() =>
                useNotieConfig(undefined, themeName),
            );

            const { customFontUrl, tocCustomFontUrl } = result.current.theme;
            expect(customFontUrl).toBeTruthy();
            expect(tocCustomFontUrl).toBeTruthy();

            // Both point at the same URL for this theme, so both links share it.
            expect(customFontUrl).toBe(tocCustomFontUrl);
            expect(getFontLinks(customFontUrl)).toHaveLength(2);

            unmount();
            expect(getFontLinks(customFontUrl)).toHaveLength(0);
        });
    });
});
