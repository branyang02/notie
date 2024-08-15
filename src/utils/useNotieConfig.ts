import { useEffect, useMemo } from "react";
import { defaultNotieConfig, NotieConfig } from "../config/NotieConfig";

export const useNotieConfig = (config?: NotieConfig) => {
    const mergedConfig = useMemo(() => {
        return { ...defaultNotieConfig, ...config };
    }, [config]);

    useEffect(() => {
        const root = document.documentElement;

        if (mergedConfig.fontSize) {
            root.style.setProperty("--blog-font-size", mergedConfig.fontSize);
        }
        if (mergedConfig.fontFamily) {
            root.style.setProperty(
                "--blog-font-family",
                mergedConfig.fontFamily,
            );
        }
    }, [mergedConfig]);

    return mergedConfig;
};
