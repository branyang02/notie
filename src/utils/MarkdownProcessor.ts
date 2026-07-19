import { NotieConfig } from "../config/NotieConfig";
import { maskProtectedRegions } from "./markdownMasking";
import { extractTocEntriesFromMasked, TocEntry } from "./toc";
import { BlockquoteMapping, EquationMapping } from "./utils";

export class MarkdownProcessor {
    private markdownContent: string;
    private config: NotieConfig;
    private equationMapping: EquationMapping = {};
    private blockquoteMapping: BlockquoteMapping = {};

    constructor(markdownContent: string, config: NotieConfig) {
        this.markdownContent = markdownContent;
        this.config = config;
    }

    public process(): {
        markdownContent: string;
        markdownSections: string[];
        equationMapping: EquationMapping;
        blockquoteMapping: BlockquoteMapping;
        tocEntries: TocEntry[];
    } {
        // Mask fenced/indented code blocks and HTML comments at the document
        // level so that section splitting, equation/blockquote scanning, and
        // heading numbering never see their contents.
        const { maskedText, unmask } = maskProtectedRegions(
            this.markdownContent,
        );

        // Normalize single-line `$$\begin{...}...\end{...}$$` display
        // environments into the canonical multi-line form BEFORE section
        // splitting and equation scanning, so the renderer (remark-math)
        // and the equation mapping agree on what is display math.
        const normalizedText =
            this.normalizeSingleLineDisplayEnvironments(maskedText);

        const sections = this.splitIntoSections(normalizedText);
        let processedSections = sections.map((section, i) => {
            section = i === 0 ? section : this.wrapInDiv(section, i); // Do not process the first section under Title
            return this.processSection(section, i);
        });

        if (this.config.theme?.numberedHeading) {
            processedSections =
                this.addHeadingNumbersToSections(processedSections);
        }

        // Extract the table of contents from the still-masked text (after
        // heading numbering, so numbered ids match what rehype-slug sees in
        // the rendered output). Reusing this pass's mask avoids the second
        // document-level maskProtectedRegions() call that
        // extractTableOfContents() would perform on the final content.
        const tocEntries = extractTocEntriesFromMasked(
            processedSections.join(""),
        );

        // Restore the original code/comment content before returning.
        const restoredSections = processedSections.map(unmask);
        for (const mapping of Object.values(this.equationMapping)) {
            mapping.equationString = unmask(mapping.equationString);
        }
        for (const mapping of Object.values(this.blockquoteMapping)) {
            mapping.blockquoteContent = unmask(mapping.blockquoteContent);
        }

        return {
            markdownContent: restoredSections.join(""),
            markdownSections: restoredSections,
            equationMapping: this.equationMapping,
            blockquoteMapping: this.blockquoteMapping,
            tocEntries,
        };
    }

    private addHeadingNumbersToSections(sections: string[]): string[] {
        const headingRegex = /^(#{2,6}) (.*)$/gm;
        const counters = [0, 0, 0, 0, 0];

        return sections.map((section) =>
            section.replace(headingRegex, (_match, hashes, title) => {
                const level = hashes.length - 2;
                counters[level]++;

                for (let i = level + 1; i < counters.length; i++) {
                    counters[i] = 0;
                }

                const numbering = counters.slice(0, level + 1).join(".");
                return `${hashes} ${numbering}&nbsp;&nbsp;&nbsp;${title}`;
            }),
        );
    }

    /**
     * Rewrites display-math environments written on a single line, e.g.
     *
     *     $$\begin{equation}\label{eq:a} y = 1 \end{equation}$$
     *
     * into the canonical multi-line form
     *
     *     $$
     *     \begin{equation}\label{eq:a} y = 1 \end{equation}
     *     $$
     *
     * remark-math v6 parses a one-line `$$...$$` as INLINE math, so KaTeX
     * rejects the display-only `equation`/`align` environments with a red
     * ParseError and never creates the `eqn-X.Y` anchor — while the
     * equation scanner still maps the `\label`, leaving `\eqref` links
     * dangling. Normalizing before section splitting and scanning keeps
     * the renderer and the mapping in agreement. Multi-line forms are
     * untouched, and this runs on MASKED text, so code blocks and HTML
     * comments can never be corrupted.
     */
    private normalizeSingleLineDisplayEnvironments(content: string): string {
        // `alignat` takes a mandatory argument (\begin{alignat}{2}); the
        // optional `\{\d+\}` after the environment name allows it without
        // disturbing the \3 back-reference that pairs \begin with \end.
        const singleLinePattern =
            /^([ \t]*)\$\$[ \t]*(\\begin\{(equation|align|gather|alignat)\}(?:\{\d+\})?.*?\\end\{\3\})[ \t]*\$\$[ \t]*$/gm;
        return content.replace(
            singleLinePattern,
            (_match, indent, body, env) => {
                let normalizedBody = body;
                if (env !== "equation") {
                    // The multi-row scanner (processAlignEnvironment) is
                    // line-based: it skips \begin/\end delimiter lines and
                    // numbers one row per line, so put the delimiters on
                    // their own lines and break rows at `\\`. This applies
                    // to every per-row environment (align, gather, alignat).
                    normalizedBody = body
                        .replace(
                            /\\begin\{(align|gather|alignat)\}(\{\d+\})?[ \t]*/,
                            "\\begin{$1}$2\n",
                        )
                        .replace(
                            /[ \t]*\\end\{(align|gather|alignat)\}$/,
                            "\n\\end{$1}",
                        )
                        .replace(/\\\\[ \t]*/g, "\\\\\n");
                }
                return `${indent}$$\n${normalizedBody}\n$$`;
            },
        );
    }

    private splitIntoSections(content: string): string[] {
        return content.split(/(?=^##\s)/gm).filter(Boolean);
    }

    private wrapInDiv(content: string, sectionIndex: number): string {
        return `<div className="sections" id="section-${sectionIndex}">\n\n${content}\n</div>\n`;
    }

    private processSection(
        sectionContent: string,
        sectionIndex: number,
    ): string {
        // The title section (index 0, everything before the first `##`) is
        // not wrapped in a `.sections` div, so the DOM labeler in Notie.tsx
        // (which only walks `.sections` divs, numbering them 1.x upward)
        // never creates `eqn-0.y` anchors or blockquote numbers for it.
        // Scanning it would produce mapping entries like `0.1` that point
        // at nonexistent anchors, so it is skipped entirely; any labels
        // found there get a console.error instead of a broken entry.
        if (sectionIndex === 0) {
            this.warnTitleSectionLabels(sectionContent);
            return sectionContent;
        }

        // Code blocks and HTML comments are already masked at the document
        // level (see process()), so the scanners below never see them.
        const finalContent = this.processEquations(
            sectionContent,
            sectionIndex,
        );
        this.processBlockquotes(sectionContent, sectionIndex);
        return finalContent;
    }

    /**
     * Warns about `\label`s inside display-math environments in the title
     * section (before the first `##` heading). Equations there render, but
     * KaTeX numbering anchors (`eqn-X.Y`) are only assigned inside
     * `.sections` divs starting at section 1, so references to these labels
     * can never resolve — the labels are reported and left unmapped.
     */
    private warnTitleSectionLabels(content: string): void {
        const equationPattern =
            /\$\$\s*\\begin\{(equation|align|gather|alignat)\}(?:\{\d+\})?[\s\S]*?\\end\{\1\}\s*\$\$/g;
        let match;
        while ((match = equationPattern.exec(content)) !== null) {
            const labelPattern = /\\label\{(.*?)\}/g;
            let labelMatch;
            while ((labelMatch = labelPattern.exec(match[0])) !== null) {
                console.error(
                    `Label "${labelMatch[1]}" is defined in the title section ` +
                        `(before the first "##" heading). Equations there are not ` +
                        `numbered (numbering starts in section 1), so references to ` +
                        `this label are unsupported and will not resolve. Move the ` +
                        `equation into a "##" section to reference it.`,
                );
            }
        }

        const blockquoteIdPattern = /<blockquote\b[^>]*\bid="([^"]+)"[^>]*>/g;
        while ((match = blockquoteIdPattern.exec(content)) !== null) {
            console.error(
                `Blockquote id "${match[1]}" is defined in the title section ` +
                    `(before the first "##" heading). Blockquotes there are not ` +
                    `numbered (numbering starts in section 1), so references to ` +
                    `this id are unsupported and will not resolve. Move the ` +
                    `blockquote into a "##" section to reference it.`,
            );
        }
    }

    private processBlockquotes(content: string, sectionIndex: number): void {
        const tagPattern = /<blockquote\b([^>]*)>/g;
        const SUPPORTED = [
            "definition",
            "theorem",
            "lemma",
            "algorithm",
            "problem",
            "proof",
            "note",
            "important",
        ];
        const counts: Record<string, number> = {};

        let tagMatch;
        while ((tagMatch = tagPattern.exec(content)) !== null) {
            const attrs = tagMatch[1];
            const classAttr = attrs.match(/\bclass="([^"]+)"/)?.[1] ?? "";
            const idAttr = attrs.match(/\bid="([^"]+)"/)?.[1];

            const type = SUPPORTED.find((t) => classAttr.includes(t));
            if (!type) continue;

            // Theorems and lemmas share a counter (mirrors DOM useEffect)
            const counterKey = type === "lemma" ? "theorem" : type;
            counts[counterKey] = (counts[counterKey] ?? 0) + 1;

            // Only store in mapping if this blockquote has a label id
            if (!idAttr) continue;

            const blockquoteNumber = `${sectionIndex}.${counts[counterKey]}`;

            const contentStart = tagMatch.index + tagMatch[0].length;
            const endIndex = content.indexOf("</blockquote>", contentStart);
            const rawContent =
                endIndex !== -1
                    ? content.slice(contentStart, endIndex).trim()
                    : "";

            if (idAttr in this.blockquoteMapping) {
                console.error(`Duplicate blockquote id found: ${idAttr}.`);
            } else {
                this.blockquoteMapping[idAttr] = {
                    blockquoteNumber,
                    blockquoteType: type,
                    blockquoteContent: rawContent,
                };
            }
        }
    }

    private processEquations(content: string, sectionIndex: number): string {
        // Matches display environments with optional whitespace between the
        // `$$` delimiters and the environment. Single-line forms have
        // already been normalized to multi-line by
        // normalizeSingleLineDisplayEnvironments(), so each equation is
        // seen exactly once here.
        // `alignat` takes a mandatory argument (\begin{alignat}{2}); the
        // optional `\{\d+\}` allows it while the \1 back-reference still
        // pairs \begin{env} with the matching \end{env}.
        const equationPattern =
            /\$\$\s*\\begin\{(equation|align|gather|alignat)\}(?:\{\d+\})?[\s\S]*?\\end\{\1\}\s*\$\$/g;
        let currEquationNumber = 1;

        let match;
        while ((match = equationPattern.exec(content)) !== null) {
            const equation = match[0];
            const env = match[1];
            if (env === "equation") {
                currEquationNumber = this.processEquationEnvironment(
                    equation,
                    sectionIndex,
                    currEquationNumber,
                );
            } else {
                // align, gather, and alignat are all numbered per row by
                // KaTeX, so they share the line-based per-row scanner.
                currEquationNumber = this.processAlignEnvironment(
                    equation,
                    sectionIndex,
                    currEquationNumber,
                );
            }
        }

        return content;
    }

    private processAlignEnvironment(
        equation: string,
        sectionIndex: number,
        currEquationNumber: number,
    ): number {
        const lines = equation.split("\n");
        let insideBlock = false;
        let blockContent = "";

        for (const line of lines) {
            if (this.isAlignEnvironmentDelimiter(line)) {
                continue; // Skip align environment delimiters
            }

            if (this.isBlockStart(line)) {
                insideBlock = true;
                blockContent = line + "\n";
            } else if (insideBlock) {
                if (this.isBlockEnd(line)) {
                    blockContent += line;
                    // KaTeX does not assign an equation number (no
                    // `.eqn-num` span) to rows containing \nonumber or
                    // \notag, so such rows must not consume a number here
                    // either — otherwise every mapping after them drifts
                    // off by one from the DOM numbering.
                    if (this.isUnnumberedLine(blockContent)) {
                        this.warnIfLabeledUnnumbered(blockContent);
                    } else {
                        this.handleLabel(
                            line,
                            blockContent,
                            sectionIndex,
                            currEquationNumber,
                        );
                        currEquationNumber++;
                    }
                    insideBlock = false;
                    blockContent = "";
                } else {
                    blockContent += line + "\n";
                }
            } else if (this.isUnnumberedLine(line)) {
                this.warnIfLabeledUnnumbered(line);
            } else {
                this.handleLabel(line, line, sectionIndex, currEquationNumber);
                currEquationNumber++;
            }
        }

        return currEquationNumber;
    }

    private isUnnumberedLine(content: string): boolean {
        return /\\nonumber\b|\\notag\b/.test(content);
    }

    private warnIfLabeledUnnumbered(content: string): void {
        const labelText = content.match(/\\label\{(.*?)\}/)?.[1];
        if (labelText) {
            console.error(
                `Label "${labelText}" is attached to an unnumbered equation line ` +
                    `(\\nonumber/\\notag). KaTeX renders no number for this line, so ` +
                    `no mapping entry was created and references to this label will ` +
                    `not resolve to a visible equation number.`,
            );
        }
    }

    private isAlignEnvironmentDelimiter(line: string): boolean {
        return (
            line.includes("$$") ||
            /\\(begin|end)\{(align|gather|alignat)\}/.test(line) ||
            /^\s*$/.test(line)
        );
    }

    private isBlockStart(line: string): boolean {
        return /\\begin{[^}]*}/.test(line);
    }

    private isBlockEnd(line: string): boolean {
        return /\\end{[^}]*}/.test(line);
    }

    private processEquationEnvironment(
        equation: string,
        sectionIndex: number,
        currEquationNumber: number,
    ): number {
        // KaTeX assigns no equation number (no `.eqn-num` span) to an
        // equation environment containing \nonumber or \notag, so it must
        // not consume a number here either — otherwise every mapping after
        // it drifts off by one from the DOM numbering (same rule as the
        // align path above).
        if (this.isUnnumberedLine(equation)) {
            this.warnIfLabeledUnnumbered(equation);
            return currEquationNumber;
        }

        const label = equation.match(/\\label\{(.*?)\}/g)?.[0];
        if (label) {
            // extract between \begin{equation} and \end{equation}
            const line = equation
                .replace(/\\label\{(.*?)\}/g, "") // remove label from equation
                .replace(/\$\$/g, "") // remove $$
                .replace(/\\begin{equation}/, "") // remove \begin{equation}
                .replace(/\\end{equation}/, ""); // remove \end{equation}
            this.handleLabel(label, line, sectionIndex, currEquationNumber);
        }
        currEquationNumber++;
        return currEquationNumber;
    }

    private handleLabel(
        line: string,
        equation: string,
        sectionIndex: number,
        currEquationNumber: number,
    ): void {
        const labelText = line.match(/\\label\{(.*?)\}/)?.[1];

        if (labelText) {
            if (labelText in this.equationMapping) {
                console.error(`Duplicate label found: ${labelText}.`);
            } else {
                const sectionLabel = `${sectionIndex}.${currEquationNumber}`;
                this.equationMapping[labelText] = {
                    equationNumber: sectionLabel,
                    equationString: equation,
                };
            }
        }
    }
}
