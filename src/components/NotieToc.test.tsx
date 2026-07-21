import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TocEntry } from "../utils/toc";
import NotieToc from "./NotieToc";

const tocTitle = "Contents";

const tocEntries: TocEntry[] = [
    { id: "first-section", level: 2, title: "First Section" },
    { id: "nested-section", level: 3, title: "Nested Section" },
    { id: "deep-section", level: 4, title: "Deep Section" },
];

function stubMatchMedia(matches: boolean) {
    vi.stubGlobal(
        "matchMedia",
        vi.fn((query: string) => ({
            matches,
            media: query,
            onchange: null,
            addListener: vi.fn(),
            removeListener: vi.fn(),
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            dispatchEvent: vi.fn(),
        })),
    );
}

const originalScrollTo = HTMLElement.prototype.scrollTo;

function stubElementScrollTo() {
    const scrollTo = vi.fn();
    // The property is defined as writable (but not configurable) in the test
    // setup, so replace it with plain assignment.
    HTMLElement.prototype.scrollTo = scrollTo;
    return scrollTo;
}

describe("NotieToc", () => {
    afterEach(() => {
        HTMLElement.prototype.scrollTo = originalScrollTo;
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });

    it("renders the title and one link per entry", () => {
        render(
            <NotieToc
                tocEntries={tocEntries}
                activeId=""
                tocTitle={tocTitle}
            />,
        );

        expect(
            screen.getByRole("navigation", { name: "Contents" }),
        ).toBeInTheDocument();
        expect(
            screen.getByRole("heading", { name: "Contents" }),
        ).toBeInTheDocument();

        const links = screen.getAllByRole("link");
        expect(links.map((link) => link.getAttribute("href"))).toEqual([
            "#first-section",
            "#nested-section",
            "#deep-section",
        ]);
    });

    it("renders the TOC title as an <h2>, not an <h1>", () => {
        render(
            <NotieToc
                tocEntries={tocEntries}
                activeId=""
                tocTitle={tocTitle}
            />,
        );

        const heading = screen.getByRole("heading", { name: "Contents" });
        expect(heading.tagName).toBe("H2");
        expect(
            screen.queryByRole("heading", { level: 1 }),
        ).not.toBeInTheDocument();
    });

    it("applies the active class only to the active entry", () => {
        render(
            <NotieToc
                tocEntries={tocEntries}
                activeId="nested-section"
                tocTitle={tocTitle}
            />,
        );

        const activeLink = screen.getByRole("link", {
            name: "Nested Section",
        });
        expect(activeLink.className).toContain("active");

        const inactiveLink = screen.getByRole("link", {
            name: "First Section",
        });
        expect(inactiveLink.className).not.toContain("active");
    });

    it("calls onNavigate with the entry id when a link is clicked", () => {
        const onNavigate = vi.fn();
        render(
            <NotieToc
                tocEntries={tocEntries}
                activeId=""
                tocTitle={tocTitle}
                onNavigate={onNavigate}
            />,
        );

        fireEvent.click(screen.getByRole("link", { name: "Deep Section" }));

        expect(onNavigate).toHaveBeenCalledTimes(1);
        expect(onNavigate).toHaveBeenCalledWith(
            "deep-section",
            expect.anything(),
        );
    });

    it("indents entries proportionally to their heading level", () => {
        render(
            <NotieToc
                tocEntries={tocEntries}
                activeId=""
                tocTitle={tocTitle}
            />,
        );

        const listItems = screen.getAllByRole("listitem");
        expect(listItems.map((item) => item.style.marginLeft)).toEqual([
            "0em",
            "1em",
            "2em",
        ]);
    });

    it("scrolls the TOC container with behavior 'auto' when reduced motion is preferred", () => {
        stubMatchMedia(true);
        const scrollTo = stubElementScrollTo();

        render(
            <NotieToc
                tocEntries={tocEntries}
                activeId="first-section"
                tocTitle={tocTitle}
            />,
        );

        expect(scrollTo).toHaveBeenCalledWith(
            expect.objectContaining({ behavior: "auto" }),
        );
    });

    it("scrolls the TOC container smoothly without a reduced motion preference", () => {
        stubMatchMedia(false);
        const scrollTo = stubElementScrollTo();

        render(
            <NotieToc
                tocEntries={tocEntries}
                activeId="first-section"
                tocTitle={tocTitle}
            />,
        );

        expect(scrollTo).toHaveBeenCalledWith(
            expect.objectContaining({ behavior: "smooth" }),
        );
    });
});
