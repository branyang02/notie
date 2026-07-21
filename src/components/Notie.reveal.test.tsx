import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Notie from "./Notie";

const SECTION_COUNT = 8;

function buildMarkdown(): string {
    let markdown = "# Demo\n\n";
    for (let i = 1; i <= SECTION_COUNT; i++) {
        markdown += `## Section ${i}

Body of section ${i}.

$$
\\begin{equation} \\label{eq:s${i}}
x_{${i}} = ${i}
\\end{equation}
$$

<blockquote class="definition" id="def:s${i}">
Definition body ${i}.
</blockquote>

`;
    }
    return markdown;
}

describe("Notie progressive reveal coalescing", () => {
    const OriginalIntersectionObserver = window.IntersectionObserver;
    let observerConstructions: number;

    beforeEach(() => {
        window.history.replaceState(null, "", "/");
        observerConstructions = 0;

        // Fake timers make the reveal cadence deterministic. On real (slow
        // CI) clocks a single section reveal can take longer than the
        // trailing debounce window, in which case every reveal legitimately
        // gets its own rescan and the coalescing assertion below would
        // depend on machine speed.
        vi.useFakeTimers();

        const Original = OriginalIntersectionObserver;
        class CountingIntersectionObserver extends Original {
            constructor(
                callback: IntersectionObserverCallback,
                options?: IntersectionObserverInit,
            ) {
                super(callback, options);
                observerConstructions++;
            }
        }

        // The setup.ts mock is defined writable but not configurable, so
        // assign directly instead of using vi.stubGlobal (defineProperty).
        window.IntersectionObserver = CountingIntersectionObserver;
        globalThis.IntersectionObserver = CountingIntersectionObserver;
    });

    afterEach(() => {
        vi.useRealTimers();
        window.IntersectionObserver = OriginalIntersectionObserver;
        globalThis.IntersectionObserver = OriginalIntersectionObserver;
    });

    it("coalesces observer rebuilds and DOM numbering across a burst of section reveals", async () => {
        const { container } = render(
            <Notie
                markdown={buildMarkdown()}
                config={{ theme: { blockquoteStyle: "latex" } }}
            />,
        );

        // Sections are revealed progressively, one per scheduler tick
        // (jsdom has no requestIdleCallback, so MarkdownRenderer falls back
        // to setTimeout(16)). Advance fake time tick by tick — faster than
        // the trailing debounce — until every section is revealed.
        const lastSectionRevealed = () =>
            screen.queryByText(`Body of section ${SECTION_COUNT}.`) !== null;
        for (let i = 0; i < SECTION_COUNT * 4 && !lastSectionRevealed(); i++) {
            await act(async () => {
                await vi.advanceTimersByTimeAsync(16);
            });
        }
        expect(lastSectionRevealed()).toBe(true);

        // Let the trailing debounce fire the single coalesced rescan.
        await act(async () => {
            await vi.advanceTimersByTimeAsync(200);
        });

        // Final numbering must be correct once the burst settles.
        expect(
            container.querySelector(`#eqn-${SECTION_COUNT}\\.1`),
        ).toHaveTextContent(`(${SECTION_COUNT}.1)`);
        expect(container.querySelector("#eqn-1\\.1")).toHaveTextContent(
            "(1.1)",
        );
        expect(
            container.querySelector(
                `[blockquote-definition-number="Definition ${SECTION_COUNT}.1"]`,
            ),
        ).not.toBeNull();
        expect(
            container.querySelector(
                '[blockquote-definition-number="Definition 1.1"]',
            ),
        ).not.toBeNull();

        // The heading IntersectionObserver must be rebuilt only a handful
        // of times (initial sync run + coalesced trailing run), not once
        // per revealed section.
        expect(observerConstructions).toBeGreaterThan(0);
        expect(observerConstructions).toBeLessThanOrEqual(3);
        expect(observerConstructions).toBeLessThan(SECTION_COUNT);
    });
});
