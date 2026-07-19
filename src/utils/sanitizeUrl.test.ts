import { describe, expect, it } from "vitest";
import { sanitizeUrl } from "./sanitizeUrl";

describe("sanitizeUrl", () => {
    it("allows http and https URLs", () => {
        expect(sanitizeUrl("http://example.com")).toBe("http://example.com");
        expect(sanitizeUrl("https://example.com/a?b=c#d")).toBe(
            "https://example.com/a?b=c#d",
        );
    });

    it("allows mailto and tel URLs", () => {
        expect(sanitizeUrl("mailto:test@example.com")).toBe(
            "mailto:test@example.com",
        );
        expect(sanitizeUrl("tel:+15555550123")).toBe("tel:+15555550123");
    });

    it("allows relative URLs and fragments", () => {
        expect(sanitizeUrl("/docs/page")).toBe("/docs/page");
        expect(sanitizeUrl("./relative")).toBe("./relative");
        expect(sanitizeUrl("relative/path")).toBe("relative/path");
        expect(sanitizeUrl("#anchor")).toBe("#anchor");
        expect(sanitizeUrl("#eqn-1.1")).toBe("#eqn-1.1");
        expect(sanitizeUrl("?query=1")).toBe("?query=1");
    });

    it("blocks javascript: URLs", () => {
        expect(sanitizeUrl("javascript:alert(1)")).toBe("");
    });

    it("blocks obfuscated javascript: URLs", () => {
        expect(sanitizeUrl("JaVaScRiPt:alert(1)")).toBe("");
        expect(sanitizeUrl(" javascript:alert(1)")).toBe("");
        expect(sanitizeUrl("\tjavascript:alert(1)")).toBe("");
        expect(sanitizeUrl("java\u0000script:alert(1)")).toBe("");
        expect(sanitizeUrl("\u0001javascript:alert(1)")).toBe("");
    });

    it("blocks vbscript: and data:text/html URLs", () => {
        expect(sanitizeUrl("vbscript:msgbox(1)")).toBe("");
        expect(sanitizeUrl("data:text/html,<script>alert(1)</script>")).toBe(
            "",
        );
        expect(
            sanitizeUrl("data:text/html;base64,PHNjcmlwdD48L3NjcmlwdD4="),
        ).toBe("");
    });

    it("blocks other unknown schemes", () => {
        expect(sanitizeUrl("file:///etc/passwd")).toBe("");
        expect(sanitizeUrl("ftp://example.com")).toBe("");
    });

    it("blocks data: URLs on links but allows data:image/* for image src", () => {
        expect(sanitizeUrl("data:image/png;base64,iVBORw0K", "href")).toBe("");
        expect(sanitizeUrl("data:image/png;base64,iVBORw0K")).toBe("");
        expect(sanitizeUrl("data:image/png;base64,iVBORw0K", "src")).toBe(
            "data:image/png;base64,iVBORw0K",
        );
        expect(sanitizeUrl("data:text/html,<script></script>", "src")).toBe("");
    });
});
