import { NotieConfig } from "../config/NotieConfig";
import { maskProtectedRegions } from "./markdownMasking";
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
    } {
        // Mask fenced/indented code blocks and HTML comments at the document
        // level so that section splitting, equation/blockquote scanning, and
        // heading numbering never see their contents.
        const { maskedText, unmask } = maskProtectedRegions(
            this.markdownContent,
        );

        const sections = this.splitIntoSections(maskedText);
        let processedSections = sections.map((section, i) => {
            section = i === 0 ? section : this.wrapInDiv(section, i); // Do not process the first section under Title
            return this.processSection(section, i);
        });

        if (this.config.theme?.numberedHeading) {
            processedSections =
                this.addHeadingNumbersToSections(processedSections);
        }

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
        // Code blocks and HTML comments are already masked at the document
        // level (see process()), so the scanners below never see them.
        const finalContent = this.processEquations(
            sectionContent,
            sectionIndex,
        );
        this.processBlockquotes(sectionContent, sectionIndex);
        return finalContent;
    }

    private processBlockquotes(content: string, sectionIndex: number): void {
        const tagPattern = /<blockquote\b([^>]*)>/g;
        const SUPPORTED = [
            "definition",
            "theorem",
            "lemma",
            "algorithm",
            "problem",
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
        // Matches both multi-line ($$\n\begin{...}...) and single-line
        // ($$\begin{...}...\end{...}$$) display environments, with optional
        // whitespace between the delimiters and the environment.
        const equationPattern =
            /\$\$\s*\\begin\{(equation|align)\}[\s\S]*?\\end\{\1\}\s*\$\$/g;
        const equations = content.match(equationPattern);
        let currEquationNumber = 1;

        if (equations) {
            equations.forEach((equation) => {
                if (equation.includes("\\begin{align}")) {
                    currEquationNumber = this.processAlignEnvironment(
                        equation,
                        sectionIndex,
                        currEquationNumber,
                    );
                } else {
                    currEquationNumber = this.processEquationEnvironment(
                        equation,
                        sectionIndex,
                        currEquationNumber,
                    );
                }
            });
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
                    this.handleLabel(
                        line,
                        blockContent,
                        sectionIndex,
                        currEquationNumber,
                    );
                    currEquationNumber++;
                    insideBlock = false;
                    blockContent = "";
                } else {
                    blockContent += line + "\n";
                }
            } else {
                this.handleLabel(line, line, sectionIndex, currEquationNumber);
                currEquationNumber++;
            }
        }

        return currEquationNumber;
    }

    private isAlignEnvironmentDelimiter(line: string): boolean {
        return (
            line.includes("$$") ||
            line.includes("\\begin{align}") ||
            line.includes("\\end{align}") ||
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
