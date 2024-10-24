export interface EquationMapping {
    [key: string]: {
        equationNumber: string;
        equationString: string;
    };
}

// Used in EquationReference.tsx
export function extractEquationInfo(
    children: Element,
    equationMapping: EquationMapping,
) {
    const href = children.getAttribute("href");
    const label = href?.split("#pre-eqn-").pop();
    if (!label) {
        throw new Error("No equation label found");
    }

    const parenthesesRemoved = children.textContent?.includes("(") ?? false;

    if (!(label in equationMapping)) {
        console.error(
            `Equation label "${label}" not found in equation mapping`,
        );

        return {
            equationNumber: `Error: reference ${label} not labeled`,
            equationString: "error",
            parenthesesRemoved: false,
        };
    }

    const { equationNumber, equationString } = equationMapping[label];

    return { equationNumber, equationString, parenthesesRemoved };
}

// Used in EquationReference.tsx
export function processEquationString(equationString: string): string {
    // equationString is a line of LaTeX equation.
    let processedEquationString = "";
    // Use `aligned` environment even though we only have one line of equation. This is because we do not have to deal with `&` symbols in the equation.
    processedEquationString += "$$\n\\begin{aligned}\n";
    processedEquationString += equationString.replace(/\\label\{[^}]*\}/g, ""); // Remove \label{...}
    processedEquationString += "\n\\end{aligned}\n$$\n";

    return processedEquationString;
}
