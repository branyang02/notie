import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CodeHeader from "./CodeHeader";

const originalClipboard = navigator.clipboard;

function stubClipboard(writeText: ReturnType<typeof vi.fn>) {
    Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        writable: true,
        value: { writeText },
    });
}

function restoreClipboard() {
    Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        writable: true,
        value: originalClipboard,
    });
}

describe("CodeHeader", () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.runOnlyPendingTimers();
        vi.useRealTimers();
        restoreClipboard();
        vi.restoreAllMocks();
    });

    it("copies the code, shows 'Copied!', and reverts after 2000ms", async () => {
        const writeText = vi.fn(() => Promise.resolve());
        stubClipboard(writeText);

        render(<CodeHeader language="python" code="print('hello')" />);

        fireEvent.click(screen.getByRole("button", { name: /copy code/i }));
        // Flush the clipboard promise's .then callback.
        await act(async () => {
            await Promise.resolve();
        });

        expect(writeText).toHaveBeenCalledWith("print('hello')");
        expect(screen.getByText("Copied!")).toBeInTheDocument();

        act(() => {
            vi.advanceTimersByTime(2000);
        });

        expect(screen.queryByText("Copied!")).not.toBeInTheDocument();
        expect(screen.getByText("Copy Code")).toBeInTheDocument();
    });

    it("announces 'Copied to clipboard' via an aria-live status region", async () => {
        const writeText = vi.fn(() => Promise.resolve());
        stubClipboard(writeText);

        render(<CodeHeader language="python" code="print('hello')" />);

        const status = screen.getByRole("status");
        expect(status).toHaveAttribute("aria-live", "polite");
        expect(status).toHaveTextContent("");

        fireEvent.click(screen.getByRole("button", { name: /copy code/i }));
        await act(async () => {
            await Promise.resolve();
        });

        expect(status).toHaveTextContent("Copied to clipboard");

        act(() => {
            vi.advanceTimersByTime(2000);
        });

        expect(status).toHaveTextContent("");
    });

    it("keeps a stable 'Copy code' accessible name while showing 'Copied!'", async () => {
        const writeText = vi.fn(() => Promise.resolve());
        stubClipboard(writeText);

        render(<CodeHeader language="python" code="print('hello')" />);

        fireEvent.click(screen.getByRole("button", { name: /copy code/i }));
        await act(async () => {
            await Promise.resolve();
        });

        // The visible label switches to "Copied!", but the accessible name
        // stays stable via aria-label.
        expect(
            screen.getByRole("button", { name: /copy code/i }),
        ).toBeInTheDocument();
        expect(screen.getByText("Copied!")).toBeInTheDocument();
    });

    it("does not show 'Copied!' when the clipboard write rejects", async () => {
        const writeText = vi.fn(() => Promise.reject(new Error("denied")));
        stubClipboard(writeText);

        const unhandled = vi.fn();
        process.on("unhandledRejection", unhandled);

        try {
            render(<CodeHeader language="python" code="print('hello')" />);

            fireEvent.click(screen.getByRole("button", { name: /copy code/i }));
            await act(async () => {
                await Promise.resolve();
                await Promise.resolve();
            });

            expect(writeText).toHaveBeenCalledWith("print('hello')");
            expect(screen.queryByText("Copied!")).not.toBeInTheDocument();
            expect(screen.getByText("Copy Code")).toBeInTheDocument();
            expect(unhandled).not.toHaveBeenCalled();
        } finally {
            process.off("unhandledRejection", unhandled);
        }
    });

    it("does nothing when navigator.clipboard is unavailable", () => {
        Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            writable: true,
            value: undefined,
        });

        render(<CodeHeader language="python" code="print('hello')" />);

        expect(() =>
            fireEvent.click(screen.getByRole("button", { name: /copy code/i })),
        ).not.toThrow();
        expect(screen.getByText("Copy Code")).toBeInTheDocument();
    });

    it("cleans up the timer on unmount without state-update warnings", async () => {
        const writeText = vi.fn(() => Promise.resolve());
        stubClipboard(writeText);

        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});

        const { unmount } = render(
            <CodeHeader language="python" code="print('hello')" />,
        );

        fireEvent.click(screen.getByRole("button", { name: /copy code/i }));
        await act(async () => {
            await Promise.resolve();
        });
        expect(screen.getByText("Copied!")).toBeInTheDocument();

        unmount();

        act(() => {
            vi.advanceTimersByTime(2000);
        });
        expect(vi.getTimerCount()).toBe(0);

        const warnings = consoleError.mock.calls
            .map((call) => String(call[0]))
            .filter(
                (message) =>
                    message.includes("act(") ||
                    message.includes("unmounted component"),
            );
        expect(warnings).toEqual([]);
    });
});
