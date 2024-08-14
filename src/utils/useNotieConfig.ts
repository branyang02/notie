import { useEffect, useMemo } from "react";
import { defaultNotieConfig, NotieConfig } from "../config/NotieConfig";

export const useNotieConfig = (config?: NotieConfig) => {
    const mergedConfig = useMemo(() => {
        return { ...defaultNotieConfig, ...config };
    }, [config]);

    useEffect(() => {
        document.documentElement.style.setProperty(
            "--show-toc",
            mergedConfig.showTableOfContents ? "20.25rem" : "0",
        );
    }, [mergedConfig.showTableOfContents]);

    return mergedConfig;
};
