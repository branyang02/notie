import { afterEach, describe, expect, it, vi } from "vitest";
import {
    BlockquoteMapping,
    EquationMapping,
    extractBlockquoteInfo,
    extractEquationInfo,
    parseExecuteLanguage,
    processEquationString,
} from "./utils";

describe("parseExecuteLanguage", () => {
    it("strips the execute- prefix", () => {
        expect(parseExecuteLanguage("execute-python")).toBe("python");
        expect(parseExecuteLanguage("execute-java")).toBe("java");
    });

    it("preserves multi-part language names", () => {
        expect(parseExecuteLanguage("execute-objective-c")).toBe("objective-c");
        expect(parseExecuteLanguage("execute-c-sharp")).toBe("c-sharp");
    });

    it("only strips a leading prefix", () => {
        expect(parseExecuteLanguage("execute-execute-python")).toBe(
            "execute-python",
        );
        expect(parseExecuteLanguage("python")).toBe("python");
    });
});

describe("extractEquationInfo", () => {
    const equationMapping: EquationMapping = {
        "eq:first": {
            equationNumber: "1.1",
            equationString: "\\begin{equation} x = 1 \\end{equation}",
        },
        "eq:k_step:q": {
            equationNumber: "2.3",
            equationString: "\\begin{equation} q = k \\end{equation}",
        },
    };

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("marks eqref: prefixed labels as parenthesesRemoved", () => {
        const result = extractEquationInfo(
            "#pre-eqn-eqref:eq:first",
            undefined,
            equationMapping,
        );
        expect(result).toEqual({
            equationNumber: "1.1",
            equationString: "\\begin{equation} x = 1 \\end{equation}",
            parenthesesRemoved: true,
        });
    });

    it("marks ref: prefixed labels as not parenthesesRemoved", () => {
        const result = extractEquationInfo(
            "#pre-eqn-ref:eq:first",
            "(1.1)", // text content must not override the explicit ref: prefix
            equationMapping,
        );
        expect(result.parenthesesRemoved).toBe(false);
        expect(result.equationNumber).toBe("1.1");
    });

    it("falls back to textContent parentheses detection for bare labels", () => {
        const withParens = extractEquationInfo(
            "#pre-eqn-eq:first",
            "(1.1)",
            equationMapping,
        );
        expect(withParens.parenthesesRemoved).toBe(true);

        const withoutParens = extractEquationInfo(
            "#pre-eqn-eq:first",
            "1.1",
            equationMapping,
        );
        expect(withoutParens.parenthesesRemoved).toBe(false);

        const noTextContent = extractEquationInfo(
            "#pre-eqn-eq:first",
            null,
            equationMapping,
        );
        expect(noTextContent.parenthesesRemoved).toBe(false);
    });

    it("preserves labels containing colons", () => {
        const result = extractEquationInfo(
            "#pre-eqn-eqref:eq:k_step:q",
            undefined,
            equationMapping,
        );
        expect(result.equationNumber).toBe("2.3");
        expect(result.equationString).toBe(
            "\\begin{equation} q = k \\end{equation}",
        );
        expect(result.parenthesesRemoved).toBe(true);
    });

    it("returns an error payload and logs for unknown labels", () => {
        const consoleErrorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});

        const result = extractEquationInfo(
            "#pre-eqn-eqref:eq:missing",
            undefined,
            equationMapping,
        );

        expect(result).toEqual({
            equationNumber: "Error: reference eq:missing not labeled",
            equationString: "error",
            parenthesesRemoved: true,
        });
        expect(consoleErrorSpy).toHaveBeenCalledWith(
            'Equation label "eq:missing" not found in equation mapping',
        );
    });

    it("throws when no label can be parsed from the href", () => {
        expect(() =>
            extractEquationInfo(undefined, undefined, equationMapping),
        ).toThrow("No equation label found");
        expect(() =>
            extractEquationInfo("#pre-eqn-", undefined, equationMapping),
        ).toThrow("No equation label found");
    });
});

describe("extractBlockquoteInfo", () => {
    const blockquoteMapping: BlockquoteMapping = {
        "def:first": {
            blockquoteNumber: "1.1",
            blockquoteType: "definition",
            blockquoteContent: "Reusable definition.",
        },
    };

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("returns the mapped blockquote info for known labels", () => {
        expect(
            extractBlockquoteInfo("#bqref-def:first", blockquoteMapping),
        ).toEqual({
            blockquoteNumber: "1.1",
            blockquoteType: "definition",
            blockquoteContent: "Reusable definition.",
        });
    });

    it("throws when the href has no label", () => {
        expect(() =>
            extractBlockquoteInfo(undefined, blockquoteMapping),
        ).toThrow("No blockquote label found");
        expect(() => extractBlockquoteInfo(null, blockquoteMapping)).toThrow(
            "No blockquote label found",
        );
        expect(() =>
            extractBlockquoteInfo("#bqref-", blockquoteMapping),
        ).toThrow("No blockquote label found");
    });

    it("returns an error payload and logs for unknown labels", () => {
        const consoleErrorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});

        expect(
            extractBlockquoteInfo("#bqref-def:missing", blockquoteMapping),
        ).toEqual({
            blockquoteNumber: "Error: reference def:missing not labeled",
            blockquoteType: "unknown",
            blockquoteContent: "error",
        });
        expect(consoleErrorSpy).toHaveBeenCalledWith(
            'Blockquote label "def:missing" not found in blockquote mapping',
        );
    });
});

describe("processEquationString", () => {
    it("strips \\label commands and wraps the equation in an aligned block", () => {
        const result = processEquationString(
            "x = 1 \\label{eq:first} + \\label{eq:second} 2",
        );
        expect(result).not.toContain("\\label");
        expect(result).toBe(
            "$$\n\\begin{aligned}\nx = 1  +  2\n\\end{aligned}\n$$\n",
        );
    });

    it("wraps label-free equations unchanged", () => {
        expect(processEquationString("E = mc^2")).toBe(
            "$$\n\\begin{aligned}\nE = mc^2\n\\end{aligned}\n$$\n",
        );
    });
});
