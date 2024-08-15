import { useEffect, useMemo } from "react";
import { FullNotieConfig, FullTheme, NotieConfig } from "../config/NotieConfig";

const starlitEclipse: FullTheme = {
    fontFamily: "'Space Grotesk', sans-serif",
    customFontUrl: "https://fonts.googleapis.com/css?family=Space%20Grotesk",
    titleColor: "rgb(243 244 246)",
    textColor: "#d1d5db",
    linkColor: "#ec4899",
    linkHoverColor: "#f472b6",
    tocColor: "#d1d5db",
    tocHoverColor: "#fffefe",
    tocUnderline: false,
    codeColor: "#6366f1",
    codeBackgroundColor: "#2d2d2d",
    backgroundColor: "rgb(3 7 18)",
    codeHeaderColor: "rgb(31 41 55)",
    staticCodeTheme: "atomOneDark",
    liveCodeTheme: "tokyoNightStorm",
    collapseSectionColor: "#353535",
    katexSize: "1.21rem",
    tableBorderColor: "#fff",
    tableBackgroundColor: "#6f6f70",
    captionColor: "#555",
    subtitleColor: "#969696",
    tikZstyle: "inverted",
};

const defaultTheme: FullTheme = {
    backgroundColor: "#333",
    fontFamily: "",
    customFontUrl: "",
    titleColor: "#fff",
    textColor: "#fff",
    linkColor: "#fff",
    linkHoverColor: "#bbb",
    tocColor: "#fff",
    tocHoverColor: "#bbb",
    tocUnderline: false,
    codeColor: "#fff",
    codeBackgroundColor: "#6e768166",
    codeHeaderColor: "rgba(175, 184, 193, 0.2)",
    staticCodeTheme: "nord",
    liveCodeTheme: "tokyoNightStorm",
    collapseSectionColor: "#444",
    katexSize: "1.21rem",
    tableBorderColor: "#ddd",
    tableBackgroundColor: "#ededed33",
    captionColor: "#fff",
    subtitleColor: "#969696",
    tikZstyle: "inverted",
};

const defaultNotieConfig: FullNotieConfig = {
    showTableOfContents: true,
    tocTitle: "Contents",
    fontSize: "1rem",
    theme: defaultTheme,
};

export function useNotieConfig(
    userConfig?: NotieConfig,
    userTheme?: "Starlit Eclipse" | "default",
): FullNotieConfig {
    const selectedTheme = useMemo(() => {
        switch (userTheme) {
            case "Starlit Eclipse":
                return starlitEclipse;
            case "default":
            default:
                return defaultTheme;
        }
    }, [userTheme]);

    const mergedConfig = useMemo(
        () => ({
            ...defaultNotieConfig,
            ...userConfig,
            theme: {
                ...selectedTheme,
                ...userConfig?.theme,
            },
        }),
        [userConfig, selectedTheme],
    );

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

        // Handle custom font
        const url = mergedConfig.theme.customFontUrl;
        if (url) {
            const fontLink = document.createElement("link");
            fontLink.href = url;
            fontLink.rel = "stylesheet";
            document.head.appendChild(fontLink);

            return () => {
                document.head.removeChild(fontLink);
            };
        }
    }, [mergedConfig]);

    return mergedConfig;
}
