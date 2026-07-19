import { act, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { fireIntersection } from "../test/setup";
import LazyRender from "./LazyRender";

describe("LazyRender", () => {
    it("renders a placeholder until an intersection is fired", () => {
        const { container } = render(
            <LazyRender minHeight={220}>
                <div data-testid="lazy-content">Hello</div>
            </LazyRender>,
        );

        // Default mock observer behavior: never intersects on its own.
        expect(screen.queryByTestId("lazy-content")).toBeNull();
        const placeholder = container.firstElementChild as HTMLElement;
        expect(placeholder.style.minHeight).toBe("220px");

        act(() => {
            fireIntersection();
        });

        expect(screen.getByTestId("lazy-content")).toBeInTheDocument();
        expect(placeholder.style.minHeight).toBe("");
    });

    it("only reveals content whose observed element intersects", () => {
        const { container } = render(
            <>
                <LazyRender>
                    <div data-testid="first">First</div>
                </LazyRender>
                <LazyRender>
                    <div data-testid="second">Second</div>
                </LazyRender>
            </>,
        );

        const [firstContainer] = Array.from(container.children);

        act(() => {
            fireIntersection(firstContainer);
        });

        expect(screen.getByTestId("first")).toBeInTheDocument();
        expect(screen.queryByTestId("second")).toBeNull();
    });

    it("ignores non-intersecting notifications", () => {
        render(
            <LazyRender>
                <div data-testid="lazy-content">Hello</div>
            </LazyRender>,
        );

        act(() => {
            fireIntersection(undefined, false);
        });

        expect(screen.queryByTestId("lazy-content")).toBeNull();
    });
});
