import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Notie from "./Notie";

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
});
