import { describe, expect, it } from "vitest";
import { parseExecuteLanguage } from "./utils";

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
