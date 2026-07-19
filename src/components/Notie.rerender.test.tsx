import { act, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MarkdownProcessor } from "../utils/MarkdownProcessor";
import Notie from "./Notie";

const markdown = `# Demo

## First Section

Some content.

\`\`\`component
{
    componentName: "Widget"
}
\`\`\`
`;

describe("Notie config identity", () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("does not reprocess markdown when a parent re-renders with an inline config object", async () => {
        const processSpy = vi.spyOn(MarkdownProcessor.prototype, "process");

        let triggerParentRender: () => void = () => {};
        const Parent = () => {
            const [, setTick] = useState(0);
            triggerParentRender = () => setTick((tick) => tick + 1);
            return (
                <Notie
                    markdown={markdown}
                    config={{
                        theme: {
                            linkColor: "#ff0000",
                            numberedHeading: true,
                        },
                    }}
                    customComponents={{
                        Widget: () => (
                            <div data-testid="custom-widget">Widget</div>
                        ),
                    }}
                />
            );
        };

        render(<Parent />);
        expect(screen.getByTestId("custom-widget")).toBeInTheDocument();
        expect(processSpy).toHaveBeenCalledTimes(1);

        // Multiple parent re-renders with a fresh inline config object each
        // time must not invalidate the merged config nor reprocess markdown.
        await act(async () => triggerParentRender());
        await act(async () => triggerParentRender());
        await act(async () => triggerParentRender());

        expect(processSpy).toHaveBeenCalledTimes(1);
        expect(screen.getByText("Some content.")).toBeInTheDocument();
    });

    it("reprocesses markdown when the config contents actually change", async () => {
        const processSpy = vi.spyOn(MarkdownProcessor.prototype, "process");

        const { rerender } = render(
            <Notie
                markdown={markdown}
                config={{ theme: { numberedHeading: false } }}
            />,
        );
        expect(processSpy).toHaveBeenCalledTimes(1);

        rerender(
            <Notie
                markdown={markdown}
                config={{ theme: { numberedHeading: true } }}
            />,
        );
        expect(processSpy).toHaveBeenCalledTimes(2);
    });
});
