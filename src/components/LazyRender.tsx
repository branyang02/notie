import React, { useEffect, useRef, useState } from "react";

const DEFAULT_ROOT_MARGIN = "600px 0px";

const LazyRender = ({
    children,
    minHeight = 160,
    rootMargin = DEFAULT_ROOT_MARGIN,
}: {
    children: React.ReactNode;
    minHeight?: number;
    rootMargin?: string;
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [shouldRender, setShouldRender] = useState(() => {
        if (typeof window === "undefined") return true;
        return !("IntersectionObserver" in window);
    });

    useEffect(() => {
        if (shouldRender) return;
        const container = containerRef.current;
        if (!container) return;

        const observer = new IntersectionObserver(
            (entries) => {
                if (entries.some((entry) => entry.isIntersecting)) {
                    setShouldRender(true);
                    observer.disconnect();
                }
            },
            { rootMargin },
        );

        observer.observe(container);
        return () => observer.disconnect();
    }, [rootMargin, shouldRender]);

    return (
        <div
            ref={containerRef}
            style={shouldRender ? undefined : { minHeight }}
        >
            {shouldRender ? children : null}
        </div>
    );
};

export default LazyRender;
