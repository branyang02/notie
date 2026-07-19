import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useShallowStableObject } from "./useShallowStableObject";

describe("useShallowStableObject", () => {
    it("reuses the previous identity when the object is shallowly equal", () => {
        const widget = () => null;
        const { result, rerender } = renderHook(
            ({ value }: { value: Record<string, unknown> | undefined }) =>
                useShallowStableObject(value),
            { initialProps: { value: { Widget: widget } } },
        );

        const first = result.current;
        rerender({ value: { Widget: widget } });
        expect(result.current).toBe(first);
    });

    it("returns a new identity when values change", () => {
        const { result, rerender } = renderHook(
            ({ value }: { value: Record<string, unknown> | undefined }) =>
                useShallowStableObject(value),
            { initialProps: { value: { Widget: () => null } } },
        );

        const first = result.current;
        // New function reference means a genuinely different object.
        rerender({ value: { Widget: () => null } });
        expect(result.current).not.toBe(first);
    });

    it("returns a new identity when keys change", () => {
        const widget = () => null;
        const { result, rerender } = renderHook(
            ({ value }: { value: Record<string, unknown> | undefined }) =>
                useShallowStableObject(value),
            {
                initialProps: {
                    value: { Widget: widget } as Record<string, unknown>,
                },
            },
        );

        const first = result.current;
        rerender({ value: { Widget: widget, Other: widget } });
        expect(result.current).not.toBe(first);
    });

    it("handles undefined values", () => {
        const initialProps: { value: Record<string, unknown> | undefined } = {
            value: undefined,
        };
        const { result, rerender } = renderHook(
            ({ value }: { value: Record<string, unknown> | undefined }) =>
                useShallowStableObject(value),
            { initialProps },
        );

        expect(result.current).toBeUndefined();
        rerender({ value: undefined });
        expect(result.current).toBeUndefined();

        const next = { Widget: () => null };
        rerender({ value: next });
        expect(result.current).toBe(next);
    });
});
