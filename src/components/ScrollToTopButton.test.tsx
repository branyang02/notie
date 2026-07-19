import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ScrollToTopButton from "./ScrollToTopButton";

function setPageYOffset(value: number) {
    Object.defineProperty(window, "pageYOffset", {
        writable: true,
        configurable: true,
        value,
    });
}

// The component coalesces scroll events through requestAnimationFrame, so we
// queue frame callbacks and flush them after each dispatched scroll event.
// (Running them synchronously inside requestAnimationFrame would break the
// component's frameRef guard: the ref reset inside the callback would be
// overwritten by the id assignment after the call returns.)
const frameCallbacks: FrameRequestCallback[] = [];

function flushAnimationFrames() {
    while (frameCallbacks.length > 0) {
        frameCallbacks.shift()!(0);
    }
}

function scrollTo(value: number) {
    act(() => {
        setPageYOffset(value);
        fireEvent.scroll(window);
        flushAnimationFrames();
    });
}

describe("ScrollToTopButton", () => {
    beforeEach(() => {
        frameCallbacks.length = 0;
        vi.spyOn(window, "requestAnimationFrame").mockImplementation(
            (callback: FrameRequestCallback) => {
                frameCallbacks.push(callback);
                return frameCallbacks.length;
            },
        );
        vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
        setPageYOffset(0);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("is hidden initially and stays hidden while scrolling down past 600", () => {
        render(<ScrollToTopButton />);
        expect(screen.queryByRole("button")).toBeNull();

        scrollTo(700);
        expect(screen.queryByRole("button")).toBeNull();

        scrollTo(1400);
        expect(screen.queryByRole("button")).toBeNull();
    });

    it("becomes visible when scrolling up while further than 600 from the top", () => {
        render(<ScrollToTopButton />);

        scrollTo(1400);
        expect(screen.queryByRole("button")).toBeNull();

        scrollTo(1200);
        expect(
            screen.getByRole("button", { name: /scroll to top/i }),
        ).toBeInTheDocument();
    });

    it("hides again when scrolling up close to the top", () => {
        render(<ScrollToTopButton />);

        scrollTo(1400);
        scrollTo(1200);
        expect(screen.getByRole("button")).toBeInTheDocument();

        scrollTo(300);
        expect(screen.queryByRole("button")).toBeNull();
    });

    it("scrolls smoothly to the top when clicked", () => {
        const scrollToSpy = vi
            .spyOn(window, "scrollTo")
            .mockImplementation(() => {});

        render(<ScrollToTopButton />);
        scrollTo(1400);
        scrollTo(1200);

        fireEvent.click(screen.getByRole("button", { name: /scroll to top/i }));

        expect(scrollToSpy).toHaveBeenCalledWith({
            top: 0,
            behavior: "smooth",
        });
    });
});
