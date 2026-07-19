import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import CodeBlock from "./CodeBlock";

// CodeMirror cannot mount in jsdom (duplicate @codemirror/state instances
// break instanceof checks), so stub the editor. The component under test is
// CodeBlock itself: its render body must not touch `navigator`.
vi.mock("@uiw/react-codemirror", () => ({
    __esModule: true,
    default: React.forwardRef<HTMLDivElement>(
        function CodeMirrorStub(_props, ref) {
            return <div ref={ref} data-testid="codemirror-stub" />;
        },
    ),
}));

// Keep the highlighter effect from resolving so no async state updates fire.
vi.mock("../utils/shikiHighlighter", () => ({
    getHighlighter: () => new Promise(() => {}),
    resolveLanguage: (language: string) => language,
}));

function overridePlatform(value: string | undefined) {
    Object.defineProperty(window.navigator, "platform", {
        configurable: true,
        get: () => value,
    });
}

function restorePlatform() {
    delete (window.navigator as unknown as Record<string, unknown>).platform;
}

describe("CodeBlock", () => {
    afterEach(() => {
        restorePlatform();
    });

    it("renders without throwing when navigator.platform is unavailable", () => {
        overridePlatform(undefined);

        expect(() =>
            render(
                <CodeBlock
                    initialCode={'print("hello")'}
                    language="python"
                    theme="github-light"
                />,
            ),
        ).not.toThrow();

        expect(screen.getByText("Run Code")).toBeInTheDocument();
        // Without platform information, the safe default is the Ctrl icon.
        expect(
            document.querySelector('[data-icon="key-control"]'),
        ).not.toBeNull();
    });

    it("shows the command icon on Mac platforms after mount", async () => {
        overridePlatform("MacIntel");

        const { container } = render(
            <CodeBlock
                initialCode={'print("hello")'}
                language="python"
                theme="github-light"
            />,
        );

        await waitFor(() => {
            expect(
                container.querySelector('[data-icon="key-command"]'),
            ).not.toBeNull();
        });
    });
});

describe("CodeBlock accessibility", () => {
    afterEach(() => {
        restorePlatform();
    });

    it("exposes an accessible name on the reset button", () => {
        render(
            <CodeBlock
                initialCode={'print("hello")'}
                language="python"
                theme="github-light"
            />,
        );

        expect(
            screen.getByRole("button", { name: /reset code/i }),
        ).toBeInTheDocument();
    });

    it("exposes an accessible name on the run button and hides the decorative key icons", () => {
        const { container } = render(
            <CodeBlock
                initialCode={'print("hello")'}
                language="python"
                theme="github-light"
            />,
        );

        expect(
            screen.getByRole("button", { name: /run code/i }),
        ).toBeInTheDocument();

        const hiddenIconGroup = container.querySelector(
            'button [aria-hidden="true"]',
        );
        expect(hiddenIconGroup).not.toBeNull();
        expect(hiddenIconGroup?.querySelectorAll("svg").length).toBeGreaterThan(
            0,
        );
    });
});
