export interface TocEntry {
    id: string;
    level: number;
    title: string;
}

export function markdownHeadingToId(title: string): string {
    return title
        .replace(/\s+/g, "-")
        .toLowerCase()
        .replace(/[+.()'`]/g, "")
        .replace(/&nbsp;/g, "")
        .replace(/&/g, "")
        .replace(/:/g, "");
}

export function extractTableOfContents(markdownContent: string): TocEntry[] {
    const entries: TocEntry[] = [];
    const pattern = /^#+ (.*)$/gm;
    let match;

    while ((match = pattern.exec(markdownContent)) !== null) {
        const [fullMatch, title] = match;
        const level = fullMatch.match(/^#+/)?.[0].length || 0;
        if (level === 1) continue;

        const cleanedTitle = title.replace(/[*]/g, "").trim();
        entries.push({
            id: markdownHeadingToId(cleanedTitle),
            level,
            title: cleanedTitle.replace(/&nbsp;/g, "\u00a0"),
        });
    }

    return entries;
}
