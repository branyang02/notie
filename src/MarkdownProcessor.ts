import { EquationMapping } from "./utils";

export class MarkdownProcessor {
    private markdownContent: string;
    private equationMapping: EquationMapping = {};

    constructor(markdownContent: string) {
        this.markdownContent = markdownContent;
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

        return {
            markdownContent: processedSections.join(""),
            equationMapping: this.equationMapping,
        };
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
        let insideBlock = false;
        let insideBlockEquation = "";

        const beginPattern = /\\begin{[^}]*}/;
        const endPattern = /\\end{[^}]*}/;
        for (const line of equation.split("\n")) {
            // Skip `$$`, `\begin{align}` and `\end{align}`
            if (
                line.includes("$$") ||
                line.includes("\\begin{align}") ||
                line.includes("\\end{align}")
            ) {
                continue;
            }
            // Check if the line matches `\begin{anything}`
            if (beginPattern.test(line)) {
                insideBlock = true; // Set the flag to indicate we are inside a block
                insideBlockEquation += line + "\n";
                continue;
            }
            // If inside a block, skip lines until we find `\end{anything}`
            if (insideBlock) {
                if (endPattern.test(line)) {
                    insideBlock = false;
                    insideBlockEquation += line;

                    this.handleLabel(
                        line,
                        insideBlockEquation,
                        sectionIndex,
                        currEquationNumber,
                    );
                    currEquationNumber++;
                    insideBlockEquation = "";
                    continue;
                }
                insideBlockEquation += line + "\n";
                continue;
            }
            this.handleLabel(line, line, sectionIndex, currEquationNumber);
            currEquationNumber++;
        }
        return currEquationNumber;
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
