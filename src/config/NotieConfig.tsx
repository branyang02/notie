export interface NotieConfig {
    showTableOfContents?: boolean;
    tocTitle?: string;
}

export const defaultNotieConfig: NotieConfig = {
    showTableOfContents: true,
    tocTitle: "Contents",
};
