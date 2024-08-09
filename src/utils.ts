function replaceReferences(
    sectionContent: string,
    equationMapping: { [key: string]: string },
): string {
    const regex = /\\(ref|eqref)\{(eq:[^}]+)\}/g;

    return sectionContent.replace(regex, (match, p1, p2) => {
        const newLabel = equationMapping[p2];

        if (newLabel) {
            return `\\${p1}{${newLabel}}`;
        } else {
            return match;
        }
    });
}

export function processSection(
    sectionContent: string,
    sectionIndex: number,
    equationMapping: { [key: string]: string },
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
                        continue;
                    }
                    // If inside a block, skip lines until we find `\end{anything}`
                    if (insideBlock) {
                        if (endPattern.test(line)) {
                            insideBlock = false;
                            if (handleLabel(line, equation)) {
                                currentEquationIndex++;
                            }
                        }
                        continue;
                    }
                    if (handleLabel(line, equation)) {
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

    // Replace the references in the content
    modifiedContent = replaceReferences(modifiedContent, equationMapping);

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
                    equationMapping[labelText] = sectionLabel;
                }
            }
        }
        return true; // No error occurred
    }
}
