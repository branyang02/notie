import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { NotieConfig } from "../config/NotieConfig";
import { TocEntry } from "../utils/toc";
import NotieToc from "./NotieToc";

const config: NotieConfig = { tocTitle: "Contents" };

const tocEntries: TocEntry[] = [
    { id: "first-section", level: 2, title: "First Section" },
    { id: "nested-section", level: 3, title: "Nested Section" },
    { id: "deep-section", level: 4, title: "Deep Section" },
];

describe("NotieToc", () => {
    it("renders the title and one link per entry", () => {
        render(
            <NotieToc tocEntries={tocEntries} activeId="" config={config} />,
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

    it("applies the active class only to the active entry", () => {
        render(
            <NotieToc
                tocEntries={tocEntries}
                activeId="nested-section"
                config={config}
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
                config={config}
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
            <NotieToc tocEntries={tocEntries} activeId="" config={config} />,
        );

        const listItems = screen.getAllByRole("listitem");
        expect(listItems.map((item) => item.style.marginLeft)).toEqual([
            "0em",
            "1em",
            "2em",
        ]);
    });
});
