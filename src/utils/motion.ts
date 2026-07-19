/**
 * Returns the scroll behavior that respects the user's reduced-motion
 * preference: "auto" when `prefers-reduced-motion: reduce` is set,
 * "smooth" otherwise. Guards `matchMedia` existence for non-browser
 * environments (e.g. jsdom or SSR).
 */
export function preferredScrollBehavior(): ScrollBehavior {
    if (
        typeof window !== "undefined" &&
        typeof window.matchMedia === "function" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
        return "auto";
    }
    return "smooth";
}
