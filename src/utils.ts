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
): string {
    console.log("Processing section", sectionContent);
    const equationMapping: { [key: string]: string } = {};
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
        /\$\$\n(?:\s*\\begin\{(equation|align|equation\*|gather\*|align\*|alignat\*)\}[\s\S]*?\n\s*\\end\{\1\}\s*\$\$)/g,
    );

    if (equations) {
        for (const equation of equations) {
            const isAlignEnvironment = equation.includes("\\begin{align}");
            const isLabeledEnvironment = !equation.includes("*");

            const labels = equation.match(/\\label\{(.*?)\}/g);

            if (labels) {
                for (const label of labels) {
                    const labelText = label.match(/\\label\{(.*?)\}/)?.[1];

                    if (labelText) {
                        if (labelText in equationMapping) {
                            modifiedContent = modifiedContent.replace(
                                equation,
                                `$\\color{red}\\text{KaTeX Error: Duplicate label: ${labelText}}$`,
                            );
                        } else {
                            const sectionLabel = `${sectionIndex}.${currentEquationIndex}`;
                            equationMapping[labelText] = sectionLabel;
                            if (isLabeledEnvironment) {
                                currentEquationIndex++;
                            }
                        }
                    }
                }
            } else if (isAlignEnvironment) {
                if (isLabeledEnvironment) {
                    currentEquationIndex++;
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
}
