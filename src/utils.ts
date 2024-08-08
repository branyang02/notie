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
    const equationMapping: { [key: string]: string } = {};
    let currentEquationIndex = 1;

    // Split the content into individual equations
    const equations = sectionContent.match(
        /\$\$\n(?:\s*\\begin\{(equation|align|equation\*|gather\*|align\*|alignat\*)\}[\s\S]*?\n\s*\\end\{\1\}\s*\$\$)/g,
    );

    if (equations) {
        for (const equation of equations) {
            // Check if the equation is in the 'align' environment
            const isAlignEnvironment = equation.includes("\\begin{align}");
            const isLabeledEnvironment = !equation.includes("*");

            // Extract the labels from the equation
            const labels = equation.match(/\\label\{(.*?)\}/g);

            if (labels) {
                for (const label of labels) {
                    const labelText = label.match(/\\label\{(.*?)\}/)?.[1];

                    if (labelText) {
                        if (labelText in equationMapping) {
                            sectionContent = sectionContent.replace(
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
                // no label, increment the index
                if (isLabeledEnvironment) {
                    currentEquationIndex++;
                }
            }
        }
    }

    // Replace the references in the content
    sectionContent = replaceReferences(sectionContent, equationMapping);

    return sectionContent;
}
