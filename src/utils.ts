export interface EquationMapping {
    [key: string]: {
        equationNumber: string;
        equationString: string;
    };
}

// Used in Notie.tsx
export function preProcessMarkdown(markdownContent: string) {
    const equationMapping: EquationMapping = {};
    const sections = splitIntoSections(markdownContent);
    const processedSections = sections.map((section, i) => {
        section = i === 0 ? section : wrapInDiv(section, i); // Do not process the first section under Title
        return processSection(section, i, equationMapping);
    });

    return {
        markdownContent: processedSections.join(""),
        equationMapping: equationMapping,
    };
}

function splitIntoSections(markdownContent: string): string[] {
    return markdownContent.split(/(?=^##\s)/gm).filter(Boolean);
}

function wrapInDiv(content: string, sectionIndex: number): string {
    return `<div className="sections" id="section-${sectionIndex}">\n\n${content}</div>\n`;
}

function processSection(
    sectionContent: string,
    sectionIndex: number,
    equationMapping: EquationMapping,
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

// Used in EquationReference.tsx
export function extractEquationInfo(
    children: Element,
    equationMapping: EquationMapping,
) {
    const equationLabel = children.textContent?.replace(/−/g, "-") || "";
    if (!equationLabel) {
        throw new Error("No equation label found");
    }

    const trimmedLabel = equationLabel.replace(/^\(|\)$/g, "");
    const parenthesesRemoved = trimmedLabel !== equationLabel;

    if (!(trimmedLabel in equationMapping)) {
        console.error(
            `Equation label "${trimmedLabel}" not found in equation mapping`,
        );

        return {
            equationNumber: `Error: reference ${trimmedLabel} not labeled`,
            equationString: "error",
            parenthesesRemoved: false,
        };
    }

    const { equationNumber, equationString } = equationMapping[trimmedLabel];

    return { equationNumber, equationString, parenthesesRemoved };
}

// Used in EquationReference.tsx
export function processEquationString(equationString: string): string {
    let processedEquationString = "";
    if (equationString.includes("\\begin{equation}")) {
        processedEquationString = equationString
            .replace(/\\label\{[^}]*\}/g, "")
            .replace(/\\begin\{align\}/g, "\\begin{aligned}")
            .replace(/\\begin\{equation\}/g, "")
            .replace(/\\end\{equation\}/g, "");
    } else {
        // We are given a single line from \begin{align}
        processedEquationString += "$$\n";
        processedEquationString += equationString
            .replace(/\\label\{[^}]*\}/g, "")
            .replace(/&=/g, "=");
        processedEquationString += "\n$$\n";
    }
    return processedEquationString;
}
