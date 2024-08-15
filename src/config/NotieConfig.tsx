import * as themes from "@uiw/codemirror-themes-all";
import * as ReactCodeBlocks from "react-code-blocks";

type LiveCodeBlockThemeNames = keyof typeof themes;
type StaticCodeBlockThemeNames = keyof typeof ReactCodeBlocks;

export interface Theme {
    backgroundColor?: CSSStyleDeclaration["backgroundColor"];
    fontFamily?: CSSStyleDeclaration["fontFamily"];
    customFontUrl?: string;
    titleColor?: CSSStyleDeclaration["color"];
    textColor?: CSSStyleDeclaration["color"];
    linkColor?: CSSStyleDeclaration["color"];
    linkHoverColor?: CSSStyleDeclaration["color"];
    tocColor?: CSSStyleDeclaration["color"];
    tocHoverColor?: CSSStyleDeclaration["color"];
    tocUnderline?: boolean;
    codeColor?: CSSStyleDeclaration["color"];
    codeBackgroundColor?: CSSStyleDeclaration["backgroundColor"];
    codeHeaderColor?: CSSStyleDeclaration["backgroundColor"];
    staticCodeTheme?: StaticCodeBlockThemeNames;
    liveCodeTheme?: LiveCodeBlockThemeNames;
    collapseSectionColor?: CSSStyleDeclaration["color"];
    katexSize?: CSSStyleDeclaration["fontSize"];
    tableBorderColor?: CSSStyleDeclaration["borderColor"];
    tableBackgroundColor?: CSSStyleDeclaration["backgroundColor"];
    captionColor?: CSSStyleDeclaration["color"];
    subtitleColor?: CSSStyleDeclaration["color"];
    tikZstyle?: "inverted" | "default";
}

export interface NotieConfig {
    showTableOfContents?: boolean;
    tocTitle?: string;
    fontSize?: CSSStyleDeclaration["fontSize"];
    theme?: Theme;
}

// FullTheme and FullNotieConfig are the types that are used in notie
export type FullTheme = Required<Theme>;

export type FullNotieConfig = Required<{
    [K in keyof NotieConfig]: K extends "theme" ? FullTheme : NotieConfig[K];
}>;
