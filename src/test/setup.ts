import "@testing-library/jest-dom/vitest";

/**
 * Configurable IntersectionObserver mock.
 *
 * Default behavior is identical to the previous inert mock: observing an
 * element never fires the callback, so components such as LazyRender keep
 * their placeholder until a test explicitly triggers an intersection via
 * `fireIntersection`.
 */
const activeIntersectionObservers = new Set<MockIntersectionObserver>();

function createIntersectionEntry(
    target: Element,
    isIntersecting: boolean,
): IntersectionObserverEntry {
    const rect = target.getBoundingClientRect();
    return {
        target,
        isIntersecting,
        intersectionRatio: isIntersecting ? 1 : 0,
        boundingClientRect: rect,
        intersectionRect: rect,
        rootBounds: null,
        time: 0,
    } as IntersectionObserverEntry;
}

class MockIntersectionObserver implements IntersectionObserver {
    readonly root = null;
    readonly rootMargin = "";
    readonly thresholds = [];

    private readonly callback: IntersectionObserverCallback;
    private readonly targets = new Set<Element>();

    constructor(callback?: IntersectionObserverCallback) {
        this.callback = callback ?? (() => {});
        activeIntersectionObservers.add(this);
    }

    disconnect() {
        this.targets.clear();
        activeIntersectionObservers.delete(this);
    }

    observe(target: Element) {
        this.targets.add(target);
    }

    takeRecords(): IntersectionObserverEntry[] {
        return [];
    }

    unobserve(target: Element) {
        this.targets.delete(target);
    }

    trigger(target: Element | undefined, isIntersecting: boolean) {
        const targets = target
            ? this.targets.has(target)
                ? [target]
                : []
            : [...this.targets];
        if (targets.length === 0) return;

        this.callback(
            targets.map((element) =>
                createIntersectionEntry(element, isIntersecting),
            ),
            this,
        );
    }
}

/**
 * Fires an intersection notification on active mock observers.
 *
 * - `target` limits the notification to observers watching that element;
 *   omit it to notify every observer about all of its observed elements.
 * - Wrap calls in `act(...)` when the observer callback updates React state.
 */
export function fireIntersection(
    target?: Element,
    isIntersecting: boolean = true,
) {
    // Copy first: callbacks may disconnect observers while we iterate.
    for (const observer of [...activeIntersectionObservers]) {
        observer.trigger(target, isIntersecting);
    }
}

Object.defineProperty(window, "IntersectionObserver", {
    writable: true,
    value: MockIntersectionObserver,
});

Object.defineProperty(globalThis, "IntersectionObserver", {
    writable: true,
    value: MockIntersectionObserver,
});

/**
 * Opt-in requestIdleCallback/cancelIdleCallback stubs.
 *
 * jsdom does not implement requestIdleCallback, and production code
 * (MarkdownRenderer.scheduleNextSection) intentionally falls back to
 * setTimeout when it is absent. The stubs are therefore NOT installed
 * globally — that would silently reroute the existing progressive-render
 * tests away from the fallback branch. Tests that want to exercise the
 * requestIdleCallback branch can opt in:
 *
 *     installIdleCallbackStub();
 *     // ...render...
 *     uninstallIdleCallbackStub();
 */
export function installIdleCallbackStub() {
    if (typeof window.requestIdleCallback === "function") return;

    Object.defineProperty(window, "requestIdleCallback", {
        writable: true,
        configurable: true,
        value: (callback: IdleRequestCallback): number =>
            window.setTimeout(
                () =>
                    callback({
                        didTimeout: false,
                        timeRemaining: () => 50,
                    }),
                0,
            ),
    });
    Object.defineProperty(window, "cancelIdleCallback", {
        writable: true,
        configurable: true,
        value: (id: number) => window.clearTimeout(id),
    });
}

export function uninstallIdleCallbackStub() {
    delete (window as { requestIdleCallback?: unknown }).requestIdleCallback;
    delete (window as { cancelIdleCallback?: unknown }).cancelIdleCallback;
}

Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    writable: true,
    value: () => {},
});

Object.defineProperty(HTMLElement.prototype, "scrollTo", {
    writable: true,
    value: () => {},
});
