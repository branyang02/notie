import { useEffect, useRef, useState } from "react";
import { ArrowUpIcon, Button, Pane } from "evergreen-ui";

const ScrollToTopButton = () => {
    const [isVisible, setIsVisible] = useState(false);
    const lastScrollTopRef = useRef(0);
    const frameRef = useRef<number | null>(null);

    const scrollToTop = () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth",
        });
    };

    useEffect(() => {
        const toggleVisibility = () => {
            if (frameRef.current !== null) return;

            frameRef.current = window.requestAnimationFrame(() => {
                const currentScrollTop = window.pageYOffset;
                const closeToTop = currentScrollTop < 600;

                setIsVisible(
                    !closeToTop && currentScrollTop < lastScrollTopRef.current,
                );
                lastScrollTopRef.current = currentScrollTop;
                frameRef.current = null;
            });
        };

        window.addEventListener("scroll", toggleVisibility, { passive: true });
        return () => {
            window.removeEventListener("scroll", toggleVisibility);
            if (frameRef.current !== null) {
                window.cancelAnimationFrame(frameRef.current);
            }
        };
    }, []);

    return isVisible ? (
        <Pane display="flex" justifyContent="center">
            <Pane
                zIndex={1000}
                position="fixed"
                top="10px"
                display="flex"
                justifyContent="center"
                maxWidth="inherit"
            >
                <Button
                    appearance="default"
                    intent="none"
                    onClick={scrollToTop}
                    iconBefore={ArrowUpIcon}
                    borderRadius={50}
                >
                    Scroll to Top
                </Button>
            </Pane>
        </Pane>
    ) : null;
};

export default ScrollToTopButton;
