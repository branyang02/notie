import { useRef } from "react";

function isShallowEqual<T extends Record<string, unknown>>(
    a: T | undefined,
    b: T | undefined,
): boolean {
    if (a === b) return true;
    if (!a || !b) return false;

    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);
    if (aKeys.length !== bKeys.length) return false;

    return aKeys.every(
        (key) => Object.prototype.hasOwnProperty.call(b, key) && a[key] === b[key],
    );
}

/**
 * Returns a referentially stable object across re-renders as long as the
 * provided object stays shallowly equal (same keys, same value references).
 *
 * Useful for props like `customComponents` whose values are functions and
 * therefore cannot be structurally serialized: consumers often pass inline
 * object literals (`customComponents={{ Widget }}`) which get a new identity
 * every render even though their contents have not changed.
 */
export function useShallowStableObject<T extends Record<string, unknown>>(
    value: T | undefined,
): T | undefined {
    const previousRef = useRef<T | undefined>(value);

    if (!isShallowEqual(previousRef.current, value)) {
        previousRef.current = value;
    }

    return previousRef.current;
}
