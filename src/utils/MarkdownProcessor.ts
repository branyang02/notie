import { NotieConfig } from "../config/NotieConfig";
import { EquationMapping } from "./utils";

export class MarkdownProcessor {
    private markdownContent: string;
    private config: NotieConfig;
    private equationMapping: EquationMapping = {};

    constructor(markdownContent: string, config: NotieConfig) {
        this.markdownContent = markdownContent;
        this.config = config;
    }

    public process(): {
        markdownContent: string;
        equationMapping: EquationMapping;
    } {
        const sections = this.splitIntoSections();
        const processedSections = sections.map((section, i) => {
            section = i === 0 ? section : this.wrapInDiv(section, i); // Do not process the first section under Title
            return this.processSection(section, i);
        });

        if (this.config.theme?.numberedHeading) {
            const processedMarkdown = this.addHeadingNumbers(
                processedSections.join(""),
            );
            return {
                markdownContent: processedMarkdown,
                equationMapping: this.equationMapping,
            };
        }

        return {
            markdownContent: processedSections.join(""),
            equationMapping: this.equationMapping,
        };
    }

    private addHeadingNumbers(markdownString: string): string {
        const headingRegex = /^(#{2,6}) (.*)$/gm;
        const counters = [0, 0, 0, 0, 0];

        return markdownString.replace(headingRegex, (_match, hashes, title) => {
            const level = hashes.length - 2;
            counters[level]++;

            for (let i = level + 1; i < counters.length; i++) {
                counters[i] = 0;
            }

            const numbering = counters.slice(0, level + 1).join(".");
            return `${hashes} ${numbering}&nbsp;&nbsp;&nbsp;${title}`;
        });
    }

    private splitIntoSections(): string[] {
        return this.markdownContent.split(/(?=^##\s)/gm).filter(Boolean);
    }

    private wrapInDiv(content: string, sectionIndex: number): string {
        return `<div className="sections" id="section-${sectionIndex}">\n\n${content}\n</div>\n`;
    }

    private processSection(
        sectionContent: string,
        sectionIndex: number,
    ): string {
        const { modifiedContent, codeBlocks } =
            this.extractCodeBlocks(sectionContent);
        const finalContent = this.processEquations(
            modifiedContent,
            sectionIndex,
        );
        return this.reinsertCodeBlocks(finalContent, codeBlocks);
    }

    private processEquations(content: string, sectionIndex: number): string {
        const equationPattern =
            /\$\$\n(?:\s*\\begin\{(equation|align)\}[\s\S]*?\n\s*\\end\{\1\}\s*\$\$)/g;
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
            line.includes("\\end{align}")
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
            this.handleLabel(label, equation, sectionIndex, currEquationNumber);
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

    private extractCodeBlocks(content: string): {
        modifiedContent: string;
        codeBlocks: string[];
    } {
        const codeBlockPattern = /```[\s\S]*?```/g;
        const codeBlocks: string[] = [];
        const modifiedContent = content.replace(codeBlockPattern, (match) => {
            codeBlocks.push(match);
            return `CODE_BLOCK_${codeBlocks.length - 1}`;
        });
        return { modifiedContent, codeBlocks };
    }

    private reinsertCodeBlocks(content: string, codeBlocks: string[]): string {
        return content.replace(
            /CODE_BLOCK_(\d+)/g,
            (_, index) => codeBlocks[Number(index)],
        );
    }
}
