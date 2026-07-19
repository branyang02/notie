import { useEffect, useMemo } from "react";
import { DEFAULT_CODE_RUNNER_URL } from "../service/api";
import {
    FullNotieConfig,
    FullTheme,
    NotieConfig,
    NotieThemes,
} from "../config/NotieConfig";

const starlitEclipse: FullTheme = {
    appearance: "dark",
    backgroundColor: "rgb(3 7 18)",
    fontFamily: "'Space Grotesk', sans-serif",
    customFontUrl: "https://fonts.googleapis.com/css?family=Space%20Grotesk",
    titleColor: "rgb(243 244 246)",
    textColor: "#d1d5db",
    linkColor: "#ec4899",
    linkHoverColor: "#f472b6",
    linkUnderline: false,
    tocFontFamily: "'Space Grotesk', sans-serif",
    tocCustomFontUrl: "https://fonts.googleapis.com/css?family=Space%20Grotesk",
    tocColor: "#d1d5db",
    tocHoverColor: "#fffefe",
    tocUnderline: false,
    codeColor: "#6366f1",
    codeBackgroundColor: "#24292e",
    codeHeaderColor: "rgb(31 41 55)",
    codeFontSize: "medium",
    codeCopyButtonHoverColor: "#8b8b8b",
    staticCodeTheme: "github-dark",
    liveCodeTheme: "github-dark",
    collapseSectionColor: "#6e6d6d87",
    katexSize: "1.21rem",
    tableBorderColor: "#fff",
    tableBackgroundColor: "#6f6f70",
    captionColor: "#555",
    subtitleColor: "#969696",
    tikZstyle: "inverted",
    blockquoteStyle: "default",
    numberedHeading: false,
    tocMarker: true,
};

const starlitEclipseLight: FullTheme = {
    appearance: "light",
    backgroundColor: "rgb(255 255 255)",
    fontFamily: "'Space Grotesk', sans-serif",
    customFontUrl: "https://fonts.googleapis.com/css?family=Space%20Grotesk",
    titleColor: "#000000",
    textColor: "#374151",
    linkColor: "#ec4899",
    linkHoverColor: "#f472b6",
    linkUnderline: false,
    tocFontFamily: "'Space Grotesk', sans-serif",
    tocCustomFontUrl: "https://fonts.googleapis.com/css?family=Space%20Grotesk",
    tocColor: "#374151",
    tocHoverColor: "#6f6f6f",
    tocUnderline: false,
    codeColor: "#6366f1",
    codeBackgroundColor: "#fafafa",
    codeHeaderColor: "rgb(232 232 232)",
    codeFontSize: "medium",
    codeCopyButtonHoverColor: "#F4F5F9",
    staticCodeTheme: "github-light",
    liveCodeTheme: "github-light",
    collapseSectionColor: "#eeeeee",
    katexSize: "1.21rem",
    tableBorderColor: "#ddd",
    tableBackgroundColor: "#f2f2f2",
    captionColor: "#555",
    subtitleColor: "#969696",
    tikZstyle: "default",
    blockquoteStyle: "default",
    numberedHeading: false,
    tocMarker: true,
};

const defaultDarkTheme: FullTheme = {
    appearance: "dark",
    backgroundColor: "#333",
    fontFamily: "",
    customFontUrl: "",
    titleColor: "#fff",
    textColor: "#fff",
    linkColor: "#4493f8",
    linkHoverColor: "#2177e8",
    linkUnderline: false,
    tocFontFamily: "",
    tocCustomFontUrl: "",
    tocColor: "#fff",
    tocHoverColor: "#bbb",
    tocUnderline: false,
    codeColor: "#fff",
    codeBackgroundColor: "#24292e",
    codeHeaderColor: "rgba(175, 184, 193, 0.2)",
    codeFontSize: "medium",
    codeCopyButtonHoverColor: "#8b8b8b",
    staticCodeTheme: "github-dark",
    liveCodeTheme: "github-dark",
    collapseSectionColor: "#444",
    katexSize: "1.21rem",
    tableBorderColor: "#ddd",
    tableBackgroundColor: "#ededed33",
    captionColor: "#fff",
    subtitleColor: "#969696",
    tikZstyle: "inverted",
    blockquoteStyle: "default",
    numberedHeading: false,
    tocMarker: true,
};

const defaultTheme: FullTheme = {
    appearance: "light",
    backgroundColor: "#fff",
    fontFamily: "",
    customFontUrl: "",
    titleColor: "#000",
    textColor: "#000",
    linkColor: "#36f",
    linkHoverColor: "#0000cf",
    linkUnderline: false,
    tocFontFamily: "",
    tocCustomFontUrl: "",
    tocColor: "#000",
    tocHoverColor: "#777",
    tocUnderline: false,
    codeColor: "#000",
    codeBackgroundColor: "#fafafa",
    codeHeaderColor: "rgba(175, 184, 193, 0.2)",
    codeFontSize: "medium",
    codeCopyButtonHoverColor: "#F4F5F9",
    staticCodeTheme: "github-light",
    liveCodeTheme: "github-light",
    collapseSectionColor: "#f0f0f0",
    katexSize: "1.21rem",
    tableBorderColor: "#ddd",
    tableBackgroundColor: "#f2f2f2",
    captionColor: "#555",
    subtitleColor: "#969696",
    tikZstyle: "default",
    blockquoteStyle: "default",
    numberedHeading: false,
    tocMarker: true,
};

// Blockquote colors per theme appearance. The light values match the
// original hardcoded colors in notie-global.css exactly; the dark values
// keep each type's hue but use darker translucent backgrounds and label
// colors that stay readable on dark page backgrounds. Text color inside
// blockquotes inherits --blog-text-color.
interface BlockquotePalette {
    types: Record<string, { bg: string; label: string }>;
    shadow: string;
}

const blockquotePalettes: Record<"light" | "dark", BlockquotePalette> = {
    light: {
        types: {
            definition: { bg: "rgba(174, 247, 126, 0.2)", label: "#31dd2e" },
            proof: { bg: "rgba(174, 247, 126, 0.2)", label: "#31dd2e" },
            equation: { bg: "rgba(126, 174, 247, 0.2)", label: "#486bd5" },
            theorem: { bg: "rgba(126, 174, 247, 0.2)", label: "#486bd5" },
            algorithm: { bg: "rgba(126, 174, 247, 0.2)", label: "#486bd5" },
            problem: { bg: "rgba(126, 174, 247, 0.2)", label: "#486bd5" },
            important: { bg: "rgba(247, 126, 126, 0.2)", label: "#dd2e2e" },
            note: {
                bg: "rgb(255 253 0 / 19%)",
                label: "lch(86 109.24 91.22)",
            },
        },
        shadow: "0 1px 2px rgba(0, 0, 0, 0.12), 0 3px 10px rgba(0, 0, 0, 0.08)",
    },
    dark: {
        types: {
            definition: { bg: "rgba(120, 220, 80, 0.14)", label: "#6ee76a" },
            proof: { bg: "rgba(120, 220, 80, 0.14)", label: "#6ee76a" },
            equation: { bg: "rgba(110, 155, 240, 0.16)", label: "#93aef2" },
            theorem: { bg: "rgba(110, 155, 240, 0.16)", label: "#93aef2" },
            algorithm: { bg: "rgba(110, 155, 240, 0.16)", label: "#93aef2" },
            problem: { bg: "rgba(110, 155, 240, 0.16)", label: "#93aef2" },
            important: { bg: "rgba(240, 100, 100, 0.16)", label: "#f57171" },
            note: { bg: "rgba(255, 234, 0, 0.12)", label: "#ffe14d" },
        },
        shadow: "0 1px 2px rgba(0, 0, 0, 0.5), 0 3px 10px rgba(0, 0, 0, 0.35)",
    },
};

const defaultNotieConfig: FullNotieConfig = {
    showTableOfContents: true,
    previewEquations: true,
    previewBlockquotes: true,
    tocTitle: "Contents",
    fontSize: "1rem",
    codeRunnerUrl: DEFAULT_CODE_RUNNER_URL,
    theme: defaultTheme,
};

export function useNotieConfig(
    userConfig?: NotieConfig,
    userTheme?: NotieThemes,
): FullNotieConfig {
    const selectedTheme = useMemo(() => {
        switch (userTheme) {
            case "Starlit Eclipse":
                return starlitEclipse;
            case "Starlit Eclipse Light":
                return starlitEclipseLight;
            case "default dark":
                return defaultDarkTheme;
            default:
                return defaultTheme;
        }
    }, [userTheme]);

    // NotieConfig is plain, JSON-serializable data (strings/booleans), so a
    // structural signature keeps the merged config referentially stable even
    // when consumers pass a new inline `config={{...}}` object every render.
    // Without this, a new config object identity would re-run the markdown
    // processing pipeline in consumers on every parent render.
    const userConfigSignature =
        userConfig === undefined ? undefined : JSON.stringify(userConfig);

    const mergedConfig = useMemo(() => {
        const structuralUserConfig: NotieConfig | undefined =
            userConfigSignature === undefined
                ? undefined
                : JSON.parse(userConfigSignature);
        return {
            ...defaultNotieConfig,
            ...structuralUserConfig,
            theme: {
                ...selectedTheme,
                ...structuralUserConfig?.theme,
            },
        };
    }, [userConfigSignature, selectedTheme]);

    useEffect(() => {
        const root = document.documentElement;

        root.style.setProperty("--blog-font-size", mergedConfig.fontSize);
        root.style.setProperty(
            "--blog-background-color",
            mergedConfig.theme.backgroundColor,
        );
        root.style.setProperty(
            "--blog-font-family",
            mergedConfig.theme.fontFamily,
        );
        root.style.setProperty(
            "--blog-title-color",
            mergedConfig.theme.titleColor,
        );
        root.style.setProperty(
            "--blog-text-color",
            mergedConfig.theme.textColor,
        );
        root.style.setProperty(
            "--blog-link-color",
            mergedConfig.theme.linkColor,
        );
        root.style.setProperty(
            "--blog-link-hover-color",
            mergedConfig.theme.linkHoverColor,
        );
        root.style.setProperty(
            "--blog-link-underline",
            mergedConfig.theme.linkUnderline ? "underline" : "none",
        );
        root.style.setProperty(
            "--note-toc-font-family",
            mergedConfig.theme.tocFontFamily,
        );
        root.style.setProperty("--note-toc-color", mergedConfig.theme.tocColor);
        root.style.setProperty(
            "--note-toc-hover-color",
            mergedConfig.theme.tocHoverColor,
        );
        root.style.setProperty(
            "--note-toc-underline",
            mergedConfig.theme.tocUnderline ? "underline" : "none",
        );
        root.style.setProperty(
            "--blog-code-color",
            mergedConfig.theme.codeColor,
        );
        root.style.setProperty(
            "--blog-code-background-color",
            mergedConfig.theme.codeBackgroundColor,
        );
        root.style.setProperty(
            "--blog-code-header-color",
            mergedConfig.theme.codeHeaderColor,
        );
        root.style.setProperty(
            "--blog-code-font-size",
            mergedConfig.theme.codeFontSize,
        );
        root.style.setProperty(
            "--blog-code-copy-button-hover-color",
            mergedConfig.theme.codeCopyButtonHoverColor,
        );
        root.style.setProperty(
            "--blog-collapse-section-color",
            mergedConfig.theme.collapseSectionColor,
        );
        root.style.setProperty(
            "--blog-katex-size",
            mergedConfig.theme.katexSize,
        );
        root.style.setProperty(
            "--blog-table-border-color",
            mergedConfig.theme.tableBorderColor,
        );
        root.style.setProperty(
            "--blog-table-background-color",
            mergedConfig.theme.tableBackgroundColor,
        );
        root.style.setProperty(
            "--blog-caption-color",
            mergedConfig.theme.captionColor,
        );
        root.style.setProperty(
            "--blog-subtitle-color",
            mergedConfig.theme.subtitleColor,
        );
        root.style.setProperty(
            "--blog-tikz-style",
            mergedConfig.theme.tikZstyle === "inverted"
                ? "invert(100%)"
                : "none",
        );
        root.style.setProperty(
            "--blog-blockquote-style",
            mergedConfig.theme.blockquoteStyle === "latex"
                ? "latex"
                : "default",
        );
        root.style.setProperty(
            "--blog-toc-marker",
            mergedConfig.theme.tocMarker ? "true" : "false",
        );

        // Theme-aware blockquote colors (backgrounds, labels, shadow).
        const blockquotePalette =
            blockquotePalettes[
                mergedConfig.theme.appearance === "dark" ? "dark" : "light"
            ];
        for (const [type, colors] of Object.entries(blockquotePalette.types)) {
            root.style.setProperty(`--blog-bq-${type}-bg`, colors.bg);
            root.style.setProperty(`--blog-bq-${type}-label`, colors.label);
        }
        root.style.setProperty("--blog-bq-shadow", blockquotePalette.shadow);

        // Handle custom fonts: append a <link> for each configured font URL
        // (main content and TOC) and remove all of them on cleanup.
        const fontUrls = [
            mergedConfig.theme.customFontUrl,
            mergedConfig.theme.tocCustomFontUrl,
        ].filter((fontUrl): fontUrl is string => Boolean(fontUrl));

        const fontLinks = fontUrls.map((fontUrl) => {
            const fontLink = document.createElement("link");
            fontLink.href = fontUrl;
            fontLink.rel = "stylesheet";
            document.head.appendChild(fontLink);
            return fontLink;
        });

        return () => {
            fontLinks.forEach((fontLink) => {
                document.head.removeChild(fontLink);
            });
        };
    }, [mergedConfig]);

    return mergedConfig;
}
