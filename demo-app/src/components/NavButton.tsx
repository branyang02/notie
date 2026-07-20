import React, { useState } from "react";
import {
    Button,
    IconButton,
    MenuIcon,
    Popover,
    Menu,
    Position,
    Text,
    Avatar,
    majorScale,
} from "evergreen-ui";
import { useNavigate } from "react-router-dom";

const NavMobileMenu = ({ tabs }: { tabs: string[] }) => {
    const navigate = useNavigate();

    const handleSelect = (tab: string) => {
        const path =
            tab === "Home" ? "/" : `/${tab.toLowerCase().replace(/\s+/g, "")}`;

        navigate(path);
    };

    return (
        <Popover
            position={Position.BOTTOM_LEFT}
            content={
                <Menu>
                    <Menu.Group>
                        {tabs.map((tab) => (
                            <Menu.Item
                                key={tab}
                                onSelect={() => handleSelect(tab)}
                            >
                                {tab}
                            </Menu.Item>
                        ))}
                    </Menu.Group>
                </Menu>
            }
        >
            {({ getRef, isShown, toggle }) => (
                <IconButton
                    // evergreen types getRef as taking a RefObject, but at
                    // runtime it receives the DOM node like a callback ref.
                    ref={
                        getRef as unknown as (
                            instance: HTMLButtonElement | null,
                        ) => void
                    }
                    icon={MenuIcon}
                    marginRight={16}
                    aria-label={
                        isShown
                            ? "Close navigation menu"
                            : "Open navigation menu"
                    }
                    aria-expanded={Boolean(isShown)}
                    aria-haspopup="menu"
                    onClick={toggle}
                />
            )}
        </Popover>
    );
};

const NavButton = ({
    label,
    darkMode,
}: {
    label: string;
    darkMode: boolean;
}) => {
    const [isHovered, setIsHovered] = useState(false);
    const navigate = useNavigate();

    const buttonStyle: React.CSSProperties = {
        backgroundColor: darkMode && isHovered ? "#444" : "transparent",
        transition: "background-color 0.3s ease",
    };

    const defaultStyle: React.CSSProperties = {
        WebkitFontSmoothing: "antialiased",
        appearance: "none",
    };

    const handleClick = () => {
        const path =
            label === "Home"
                ? "/"
                : `/${label.toLowerCase().replace(/\s+/g, "")}`;

        navigate(path);
    };

    return (
        <Button
            appearance="minimal"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onClick={handleClick}
            style={darkMode ? buttonStyle : defaultStyle}
        >
            <Text size={500} color={darkMode ? "tint2" : "default"}>
                {label}
            </Text>
        </Button>
    );
};

const GitHubButton = ({ darkMode }: { darkMode: boolean }) => {
    const [isHovered, setIsHovered] = useState(false);

    const buttonStyle: React.CSSProperties = {
        backgroundColor: darkMode && isHovered ? "#444" : "transparent",
        transition: "background-color 0.3s ease",
    };

    const defaultStyle: React.CSSProperties = {
        WebkitFontSmoothing: "antialiased",
        appearance: "none",
    };

    const handleClick = () => {
        window.open("https://github.com/branyang02/notie", "_blank");
    };

    return (
        <IconButton
            appearance="minimal"
            aria-label="GitHub repository"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onClick={handleClick}
            style={darkMode ? buttonStyle : defaultStyle}
            icon={
                <Avatar
                    src={"/github-mark.svg"}
                    size={majorScale(3)}
                    style={{
                        filter: darkMode ? "invert(1)" : "none",
                        cursor: "pointer",
                    }}
                />
            }
        />
    );
};

export { GitHubButton, NavButton, NavMobileMenu };
