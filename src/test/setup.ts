import "@testing-library/jest-dom/vitest";

class MockIntersectionObserver implements IntersectionObserver {
    readonly root = null;
    readonly rootMargin = "";
    readonly thresholds = [];

    disconnect() {}
    observe() {}
    takeRecords(): IntersectionObserverEntry[] {
        return [];
    }
    unobserve() {}
}

Object.defineProperty(window, "IntersectionObserver", {
    writable: true,
    value: MockIntersectionObserver,
});

Object.defineProperty(globalThis, "IntersectionObserver", {
    writable: true,
    value: MockIntersectionObserver,
});

Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    writable: true,
    value: () => {},
});

Object.defineProperty(HTMLElement.prototype, "scrollTo", {
    writable: true,
    value: () => {},
});
