/**
 * Document-level masking of "protected" markdown regions (fenced code
 * blocks, indented code blocks, and HTML comments) so that downstream
 * regex-based processing (section splitting, equation/blockquote scanning,
 * heading numbering, table-of-contents extraction) never sees their
 * contents.
 *
 * Each protected region is replaced with a unique placeholder token that:
 * - cannot collide with document content (the token base is grown until it
 *   does not occur in the source text),
 * - never starts a line with `#` or contains backticks, so it survives
 *   section splitting and is not transformed by later regexes.
 */

export interface MaskResult {
    /** The document with all protected regions replaced by tokens. */
    maskedText: string;
    /** Restores the original text for every token found in the input. */
    unmask: (text: string) => string;
}

const isBlank = (line: string): boolean => line.trim() === "";

const isIndentedCodeLine = (line: string): boolean =>
    /^(?: {4}|\t)/.test(line) && !isBlank(line);

const indentationColumns = (line: string): number => {
    let columns = 0;
    for (const char of line) {
        if (char === " ") {
            columns++;
            continue;
        }
        if (char === "\t") {
            columns += 4;
            continue;
        }
        break;
    }
    return columns;
};

const prefixColumns = (text: string): number => {
    let columns = 0;
    for (const char of text) {
        columns += char === "\t" ? 4 : 1;
    }
    return columns;
};

const listMarkerMatch = (
    line: string,
): { markerIndent: number; contentIndent: number } | null => {
    const match = line.match(/^([ \t]*)(?:[-+*]|\d+[.)])([ \t]+)/);
    if (!match) return null;
    return {
        markerIndent: indentationColumns(match[1]),
        contentIndent: prefixColumns(match[0]),
    };
};

const listContinuationContext = (
    lines: string[],
    index: number,
): { contentIndent: number } | null => {
    const currentIndent = indentationColumns(lines[index]);

    const findLazyContinuationMarker = (
        startIndex: number,
    ): { contentIndent: number } | null => {
        for (let i = startIndex; i >= 0; i--) {
            const line = lines[i];
            if (isBlank(line)) return null;

            const marker = listMarkerMatch(line);
            if (marker && marker.markerIndent < currentIndent) {
                return { contentIndent: marker.contentIndent };
            }
        }

        return null;
    };

    for (let i = index - 1; i >= 0; i--) {
        const line = lines[i];
        if (isBlank(line)) continue;

        const marker = listMarkerMatch(line);
        if (marker && marker.markerIndent < currentIndent) {
            return { contentIndent: marker.contentIndent };
        }

        if (indentationColumns(line) < currentIndent) {
            return findLazyContinuationMarker(i - 1);
        }
    }

    return null;
};

const isIndentedCodeBlockStart = (lines: string[], index: number): boolean => {
    const line = lines[index];
    if (!isIndentedCodeLine(line)) return false;

    const listContext = listContinuationContext(lines, index);
    if (!listContext) return true;

    return indentationColumns(line) >= listContext.contentIndent + 4;
};

export function maskProtectedRegions(text: string): MaskResult {
    // Grow the token base until it cannot collide with document content.
    let base = "NOTIEMASK";
    while (text.includes(base)) {
        base += "X";
    }

    const store: string[] = [];
    const makeToken = (original: string): string => {
        store.push(original);
        return `${base}${store.length - 1}${base}`;
    };

    const lines = text.split("\n");
    const out: string[] = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];

        // Fenced code block: opening fence of 3+ backticks or tildes at line
        // start (up to 3 spaces of indentation) with an optional info string.
        // Per CommonMark, a backtick fence's info string may not contain a
        // backtick (that would be an inline code span instead).
        const fenceMatch = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
        const isFenceOpen =
            fenceMatch !== null &&
            !(fenceMatch[1][0] === "`" && fenceMatch[2].includes("`"));

        if (isFenceOpen && fenceMatch) {
            const fenceChar = fenceMatch[1][0];
            const fenceLength = fenceMatch[1].length;
            // Closing fence: same character, at least the same length, at
            // line start, nothing but whitespace after it.
            const closeRe = new RegExp(
                `^ {0,3}[${fenceChar}]{${fenceLength},}\\s*$`,
            );

            const block: string[] = [line];
            i++;
            // An unterminated fence runs to EOF.
            while (i < lines.length) {
                block.push(lines[i]);
                i++;
                if (closeRe.test(block[block.length - 1])) {
                    break;
                }
            }
            out.push(makeToken(block.join("\n")));
            continue;
        }

        // Indented code block (pragmatic CommonMark subset): consecutive
        // lines indented >= 4 spaces (or a tab), starting after a blank line
        // (or at the start of the document). Interior blank lines are part
        // of the block only when followed by another indented line.
        const prevBlank = out.length === 0 || isBlank(out[out.length - 1]);
        if (isIndentedCodeBlockStart(lines, i) && prevBlank) {
            const block: string[] = [line];
            i++;
            while (i < lines.length) {
                if (isIndentedCodeBlockStart(lines, i)) {
                    block.push(lines[i]);
                    i++;
                    continue;
                }
                if (isBlank(lines[i])) {
                    let j = i;
                    while (j < lines.length && isBlank(lines[j])) {
                        j++;
                    }
                    if (
                        j < lines.length &&
                        isIndentedCodeBlockStart(lines, j)
                    ) {
                        while (i < j) {
                            block.push(lines[i]);
                            i++;
                        }
                        continue;
                    }
                }
                break;
            }
            out.push(makeToken(block.join("\n")));
            continue;
        }

        out.push(line);
        i++;
    }

    // HTML comments (well-formed <!-- ... -->, possibly spanning lines).
    // Comments inside code regions are already hidden behind tokens, and
    // tokens themselves contain no "<!--", so this is safe on the joined
    // text.
    const masked = out
        .join("\n")
        .replace(/<!--[\s\S]*?-->/g, (match) => makeToken(match));

    const tokenRe = new RegExp(`${base}(\\d+)${base}`, "g");
    const unmask = (input: string): string =>
        // Stored originals cannot contain the token base, so one pass is
        // enough (no nested tokens can survive in the output).
        input.replace(tokenRe, (_match, index) => store[Number(index)]);

    return { maskedText: masked, unmask };
}
