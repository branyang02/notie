export function preProcessMarkdown(
    markdownContent: string,
    equationMapping: {
        [key: string]: {
            equationNumber: string;
            equationString: string;
        };
    },
): string {
    const pattern = /^(```(\w+)|## .+)$/gm;
    const parts: string[] = [];

    let lastIndex = 0;
    let sectionIndex = 0;
    let currentSectionContent = "";

    markdownContent.replace(pattern, (match, _p1, p2, offset) => {
        if (sectionIndex > 0) {
            currentSectionContent += markdownContent.slice(lastIndex, offset);
        } else {
            parts.push(markdownContent.slice(lastIndex, offset));
        }

        if (p2) {
            // Code block
            currentSectionContent += `\`\`\`language-${p2}`;
        } else {
            // Add section dividers
            if (sectionIndex > 0) {
                currentSectionContent += `</div>\n`;
                parts.push(
                    processSection(
                        currentSectionContent,
                        sectionIndex,
                        equationMapping,
                    ),
                );
                currentSectionContent = "";
            }
            sectionIndex++;
            currentSectionContent += `<div className="sections" id="section-${sectionIndex}">\n\n${match}\n`;
        }

        lastIndex = offset + match.length;
        return match;
    });

    currentSectionContent += markdownContent.slice(lastIndex);

    if (sectionIndex > 0) {
        currentSectionContent += "</div>\n";
        parts.push(
            processSection(
                currentSectionContent,
                sectionIndex,
                equationMapping,
            ),
        );
    } else {
        parts.push(currentSectionContent);
    }

    return parts.join("");
}

function processSection(
    sectionContent: string,
    sectionIndex: number,
    equationMapping: {
        [key: string]: {
            equationNumber: string;
            equationString: string;
        };
    },
): string {
    let currentEquationIndex = 1;

    // Extract content wrapped in triple backticks to exclude it from processing
    const codeBlockPattern = /```[\s\S]*?```/g;
    const codeBlocks: string[] = [];
    let modifiedContent = sectionContent;

    // Replace code blocks with placeholders
    modifiedContent = modifiedContent.replace(codeBlockPattern, (match) => {
        codeBlocks.push(match);
        return `CODE_BLOCK_${codeBlocks.length - 1}`;
    });

    // Split the content into individual equations
    const equations = modifiedContent.match(
        /\$\$\n(?:\s*\\begin\{(equation|align)\}[\s\S]*?\n\s*\\end\{\1\}\s*\$\$)/g,
    );

    if (equations) {
        for (const equation of equations) {
            const isAlignEnvironment = equation.includes("\\begin{align}");

            if (isAlignEnvironment) {
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
                            if (handleLabel(line, insideBlockEquation)) {
                                currentEquationIndex++;
                            }
                            insideBlockEquation = "";
                            continue;
                        }
                        insideBlockEquation += line + "\n";
                        continue;
                    }
                    if (handleLabel(line, line)) {
                        currentEquationIndex++;
                    }
                }
            } else {
                // We are in `/begin{equation}` environment, only one label is allowed
                const label = equation.match(/\\label\{(.*?)\}/g)?.[0];
                if (label) {
                    if (handleLabel(label, equation)) {
                        currentEquationIndex++;
                    }
                }
            }
        }
    }

    // Reinsert the code blocks
    modifiedContent = modifiedContent.replace(
        /CODE_BLOCK_(\d+)/g,
        (_, index) => {
            return codeBlocks[Number(index)];
        },
    );

    return modifiedContent;

    function handleLabel(line: string, equation: string): boolean {
        if (line.includes("\\label")) {
            const labelText = line.match(/\\label\{(.*?)\}/)?.[1];

            if (labelText) {
                if (labelText in equationMapping) {
                    modifiedContent = modifiedContent.replace(
                        equation,
                        `$\\color{red}\\text{KaTeX Error: Duplicate label: ${labelText}}$`,
                    );
                    return false; // Error occurred
                } else {
                    const sectionLabel = `${sectionIndex}.${currentEquationIndex}`;
                    equationMapping[labelText] = {
                        equationNumber: sectionLabel,
                        equationString: equation,
                    };
                }
            }
        }
        return true; // No error occurred
    }
}
