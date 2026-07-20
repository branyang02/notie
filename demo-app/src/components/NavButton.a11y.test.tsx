import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";
import { GitHubButton, NavMobileMenu } from "./NavButton";

// The demo app resolves react-router-dom from its own node_modules, which
// bundles a second React copy that breaks hooks under the root test runner.
// The components under test only consume useNavigate, so mock it.
vi.mock("react-router-dom", () => ({
    useNavigate: () => vi.fn(),
}));

describe("demo navbar icon buttons accessibility", () => {
    test("GitHub icon button has an accessible name", () => {
        render(<GitHubButton darkMode={false} />);
        expect(
            screen.getByRole("button", { name: "GitHub repository" }),
        ).toBeInTheDocument();
    });

    test("mobile nav menu button has an accessible name and exposes state", () => {
        render(<NavMobileMenu tabs={["Home", "Examples"]} />);

        const closedButton = screen.getByRole("button", {
            name: "Open navigation menu",
        });
        expect(closedButton).toHaveAttribute("aria-expanded", "false");
        expect(closedButton).toHaveAttribute("aria-haspopup", "menu");

        fireEvent.click(closedButton);

        const openButton = screen.getByRole("button", {
            name: "Close navigation menu",
        });
        expect(openButton).toHaveAttribute("aria-expanded", "true");
    });
});
