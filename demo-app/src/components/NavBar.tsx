import { Pane, Heading, majorScale, Combobox, Text } from "evergreen-ui";
import { useTheme } from "../context/useTheme";
import { useEffect, useState } from "react";
import { GitHubButton, NavButton, NavMobileMenu } from "./NavButton";
import { useNavigate } from "react-router-dom";

const NavBar = () => {
    const version = import.meta.env.PACKAGE_VERSION;
    const { theme, setTheme } = useTheme();
    const [isMobile, setIsMobile] = useState(window.innerWidth <= 1000);
    const navigate = useNavigate();
    const darkMode = theme === "default dark" || theme === "Starlit Eclipse";

    const NAME = "notie";
    const TABS = ["Home", "Examples", "Tutorial", "Contribute"];

    const handleResize = () => {
        setIsMobile(window.innerWidth <= 1000);
    };

    useEffect(() => {
        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
        };
    }, []);

    return (
        <Pane
            display="flex"
            alignItems="center"
            justifyContent="center"
            height={majorScale(10)}
            borderBottom="default"
        >
            <Pane
                display="flex"
                alignItems="center"
                justifyContent="space-between"
                width="100%"
                maxWidth={majorScale(180)}
                padding={majorScale(3)}
            >
                <Heading
                    size={800}
                    color={darkMode ? "white" : "default"}
                    style={{ cursor: "pointer" }}
                    onClick={() => navigate("/")}
                >
                    <Pane>
                        {NAME}
                        <Text display="block" color="muted">
                            v{version}
                        </Text>
                    </Pane>
                </Heading>

                <Pane display="flex" alignItems="center">
                    <Pane
                        display="flex"
                        alignItems="center"
                        marginRight={majorScale(1)}
                    >
                        <Combobox
                            initialSelectedItem={{ label: theme }}
                            width={majorScale(20)}
                            items={[
                                { label: "default" },
                                { label: "default dark" },
                                { label: "Starlit Eclipse" },
                                { label: "Starlit Eclipse Light" },
                            ]}
                            itemToString={(item) => (item ? item.label : "")}
                            onChange={(selected) => setTheme(selected.label)}
                        />
                    </Pane>
                    {isMobile ? (
                        <NavMobileMenu tabs={TABS} />
                    ) : (
                        TABS.map((label) => (
                            <NavButton
                                key={label}
                                label={label}
                                darkMode={darkMode}
                            />
                        ))
                    )}
                    <GitHubButton darkMode={darkMode} />
                </Pane>
            </Pane>
        </Pane>
    );
};

export default NavBar;
