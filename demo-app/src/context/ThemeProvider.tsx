import React, { createContext, ReactNode, useEffect, useState } from "react";
import { NotieThemes } from "notie-markdown";

type ThemeContextType = {
    theme: NotieThemes;
    setTheme: (theme: NotieThemes) => void;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: ReactNode }> = ({
    children,
}) => {
    const [theme, setTheme] = useState<NotieThemes>(() => {
        const savedTheme = localStorage.getItem("theme") as NotieThemes | null;
        if (savedTheme && isValidTheme(savedTheme)) {
            return savedTheme;
        }
        return getSystemTheme();
    });

    useEffect(() => {
        localStorage.setItem("theme", theme);
    }, [theme]);

    return (
        <ThemeContext.Provider value={{ theme, setTheme }}>
            {children}
        </ThemeContext.Provider>
    );
};

function isValidTheme(theme: string): theme is NotieThemes {
    return [
        "default",
        "default dark",
        "Starlit Eclipse",
        "Starlit Eclipse Light",
    ].includes(theme);
}

function getSystemTheme(): NotieThemes {
    return window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "default dark"
        : "default";
}

export { ThemeContext };
