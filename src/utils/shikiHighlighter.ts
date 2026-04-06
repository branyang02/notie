import { createHighlighterCore } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import type { BundledTheme } from "shiki";

export const LANGUAGE_MAP: Record<string, string> = {
    python: "python",
    py: "python",
    javascript: "javascript",
    js: "javascript",
    typescript: "typescript",
    ts: "typescript",
    tsx: "tsx",
    jsx: "jsx",
    cpp: "cpp",
    "c++": "cpp",
    c: "c",
    go: "go",
    java: "java",
    rust: "rust",
    html: "html",
    css: "css",
    json: "json",
    bash: "bash",
    sh: "bash",
    shell: "bash",
    markdown: "markdown",
    md: "markdown",
    yaml: "yaml",
    toml: "toml",
    sql: "sql",
};

export function resolveLanguage(lang: string): string {
    return LANGUAGE_MAP[lang?.toLowerCase()] ?? "text";
}

export const PRELOADED_THEMES: BundledTheme[] = [
    "github-dark",
    "github-light",
    "nord",
    "one-dark-pro",
    "one-light",
    "tokyo-night",
];

let highlighterPromise: ReturnType<typeof createHighlighterCore> | null = null;

export function getHighlighter() {
    if (!highlighterPromise) {
        highlighterPromise = createHighlighterCore({
            langs: [
                import("@shikijs/langs/python"),
                import("@shikijs/langs/javascript"),
                import("@shikijs/langs/typescript"),
                import("@shikijs/langs/tsx"),
                import("@shikijs/langs/jsx"),
                import("@shikijs/langs/cpp"),
                import("@shikijs/langs/c"),
                import("@shikijs/langs/go"),
                import("@shikijs/langs/java"),
                import("@shikijs/langs/rust"),
                import("@shikijs/langs/html"),
                import("@shikijs/langs/css"),
                import("@shikijs/langs/json"),
                import("@shikijs/langs/bash"),
                import("@shikijs/langs/markdown"),
                import("@shikijs/langs/yaml"),
                import("@shikijs/langs/toml"),
                import("@shikijs/langs/sql"),
            ],
            themes: [
                import("@shikijs/themes/github-dark"),
                import("@shikijs/themes/github-light"),
                import("@shikijs/themes/nord"),
                import("@shikijs/themes/one-dark-pro"),
                import("@shikijs/themes/one-light"),
                import("@shikijs/themes/tokyo-night"),
            ],
            engine: createJavaScriptRegexEngine(),
        });
    }
    return highlighterPromise;
}
