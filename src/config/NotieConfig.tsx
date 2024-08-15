export interface NotieConfig {
    showTableOfContents?: boolean;
    tocTitle?: string;
    fontSize?: CSSStyleDeclaration["fontSize"];
    fontFamily?: CSSStyleDeclaration["fontFamily"];
}

export const defaultNotieConfig: NotieConfig = {
    showTableOfContents: true,
    tocTitle: "Contents",
};
