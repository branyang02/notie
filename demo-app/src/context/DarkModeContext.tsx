import React, { createContext, ReactNode, useEffect, useState } from "react";

type DarkModeContextType = {
    darkMode: boolean;
    toggleDarkMode: () => void;
};

const DarkModeContext = createContext<DarkModeContextType | undefined>(
    undefined,
);

export const DarkModeProvider: React.FC<{ children: ReactNode }> = ({
    children,
}) => {
    const [darkMode, setDarkMode] = useState<boolean>(() => {
        const savedMode = localStorage.getItem("darkMode");
        if (savedMode === null) {
            const currentHour = new Date().getHours();
            return currentHour >= 18 || currentHour < 6;
        }
        return savedMode === "true";
    });

    useEffect(() => {
        localStorage.setItem("darkMode", darkMode.toString());
    }, [darkMode]);

    const toggleDarkMode = () => {
        setDarkMode((prevMode) => !prevMode);
    };

    return (
        <DarkModeContext.Provider value={{ darkMode, toggleDarkMode }}>
            {children}
        </DarkModeContext.Provider>
    );
};

export { DarkModeContext };
