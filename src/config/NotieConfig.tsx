import * as liveThemes from "@uiw/codemirror-themes-all";
import * as staticThemes from "react-syntax-highlighter/dist/esm/styles/hljs";

type LiveCodeBlockThemeNames = keyof typeof liveThemes;
type StaticCodeBlockThemeNames = keyof typeof staticThemes;

// Theme Type with all optional properties
export interface Theme {
    appearance?: "light" | "dark";
    backgroundColor?: CSSStyleDeclaration["backgroundColor"];
    fontFamily?: CSSStyleDeclaration["fontFamily"];
    customFontUrl?: string;
    titleColor?: CSSStyleDeclaration["color"];
    textColor?: CSSStyleDeclaration["color"];
    linkColor?: CSSStyleDeclaration["color"];
    linkHoverColor?: CSSStyleDeclaration["color"];
    linkUnderline?: boolean;
    tocColor?: CSSStyleDeclaration["color"];
    tocHoverColor?: CSSStyleDeclaration["color"];
    tocUnderline?: boolean;
    codeColor?: CSSStyleDeclaration["color"];
    codeBackgroundColor?: CSSStyleDeclaration["backgroundColor"];
    codeHeaderColor?: CSSStyleDeclaration["backgroundColor"];
    codeCopyButtonHoverColor?: CSSStyleDeclaration["color"];
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

// NotieConfig Type with all optional properties
export interface NotieConfig {
    showTableOfContents?: boolean;
    tocTitle?: string;
    fontSize?: CSSStyleDeclaration["fontSize"];
    theme?: Theme;
}

// NotieThemes consists of the string literals of the predefined themes
export type NotieThemes =
    | "default"
    | "default dark"
    | "Starlit Eclipse"
    | "Starlit Eclipse Light";

// FullTheme and FullNotieConfig are the types that are used in notie
export type FullTheme = Required<Theme>;
export type FullNotieConfig = Required<{
    [K in keyof NotieConfig]: K extends "theme" ? FullTheme : NotieConfig[K];
}>;
