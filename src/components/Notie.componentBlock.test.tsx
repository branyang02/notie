import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Notie from "./Notie";
import type { CustomComponentProps } from "../config/NotieConfig";
import {
    extractTableOfContents,
    sanitizeUrl,
    type CustomComponents,
    type TocEntry,
} from "../index";

describe("Notie component code blocks", () => {
    it("renders an alert instead of crashing when the component JSON is invalid", () => {
        // The URL value contains a colon, so the naive key-quoting regex
        // produces invalid JSON ("https"::) and JSON.parse throws.
        expect(() =>
            render(
                <Notie
                    markdown={`# Demo

\`\`\`component
{
    componentName: "Widget",
    url: "https://example.com"
}
\`\`\`
`}
                    customComponents={{
                        Widget: () => (
                            <div data-testid="custom-widget">Widget</div>
                        ),
                    }}
                />,
            ),
        ).not.toThrow();

        expect(
            screen.getByText(/Invalid component configuration/i),
        ).toBeInTheDocument();
        expect(screen.queryByTestId("custom-widget")).not.toBeInTheDocument();
    });

    it("renders an alert when the component JSON has no componentName", () => {
        render(
            <Notie
                markdown={`# Demo

\`\`\`component
{ "some": "value" }
\`\`\`
`}
                customComponents={{
                    Widget: () => <div data-testid="custom-widget">Widget</div>,
                }}
            />,
        );

        expect(
            screen.getByText(/Invalid component configuration/i),
        ).toBeInTheDocument();
    });

    it("still renders valid custom components", () => {
        render(
            <Notie
                markdown={`# Demo

\`\`\`component
{
    componentName: "Widget"
}
\`\`\`
`}
                customComponents={{
                    Widget: () => <div data-testid="custom-widget">Widget</div>,
                }}
            />,
        );

        expect(screen.getByTestId("custom-widget")).toHaveTextContent("Widget");
    });

    it("passes the parsed component block JSON to the component as `config`", () => {
        const seenConfigs: Array<CustomComponentProps["config"]> = [];
        const Widget = ({ config }: CustomComponentProps) => {
            seenConfigs.push(config);
            return (
                <div data-testid="configured-widget">
                    {String(config?.label)}
                </div>
            );
        };
        const customComponents: CustomComponents = { Widget };

        render(
            <Notie
                markdown={`# Demo

\`\`\`component
{
    componentName: "Widget",
    label: "hello",
    count: 3,
    nested: { flag: true }
}
\`\`\`
`}
                customComponents={customComponents}
            />,
        );

        expect(screen.getByTestId("configured-widget")).toHaveTextContent(
            "hello",
        );
        expect(seenConfigs.length).toBeGreaterThan(0);
        expect(seenConfigs[0]).toEqual({
            componentName: "Widget",
            label: "hello",
            count: 3,
            nested: { flag: true },
        });
    });

    it("still renders zero-prop legacy components", () => {
        render(
            <Notie
                markdown={`# Demo

\`\`\`component
{
    componentName: "Legacy",
    ignored: "value"
}
\`\`\`
`}
                customComponents={{
                    Legacy: () => <div data-testid="legacy-widget">Legacy</div>,
                }}
            />,
        );

        expect(screen.getByTestId("legacy-widget")).toHaveTextContent("Legacy");
    });

    it("exposes sanitizeUrl and extractTableOfContents from the public API", () => {
        expect(sanitizeUrl("javascript:alert(1)")).toBe("");
        expect(sanitizeUrl("https://example.com/")).toBe(
            "https://example.com/",
        );

        const entries: TocEntry[] = extractTableOfContents(
            "# Title\n\n## Section",
        );
        expect(entries).toEqual([
            { id: "section", level: 2, title: "Section" },
        ]);
    });
});
