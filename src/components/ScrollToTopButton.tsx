import { useEffect, useRef, useState } from "react";
import { ArrowUpIcon, Button, Pane } from "evergreen-ui";
import { preferredScrollBehavior } from "../utils/motion";

const ScrollToTopButton = () => {
    const [isVisible, setIsVisible] = useState(false);
    const lastScrollTopRef = useRef(0);
    const frameRef = useRef<number | null>(null);

    const scrollToTop = () => {
        window.scrollTo({
            top: 0,
            behavior: preferredScrollBehavior(),
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

    // Keep the button mounted and toggle visibility with CSS so keyboard
    // focus is not lost when it hides. `visibility: hidden` also removes the
    // button from the tab order while hidden.
    return (
        <Pane
            display="flex"
            justifyContent="center"
            style={{
                visibility: isVisible ? "visible" : "hidden",
                opacity: isVisible ? 1 : 0,
                transition: "opacity 0.2s ease, visibility 0.2s ease",
            }}
            aria-hidden={!isVisible}
        >
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
                    tabIndex={isVisible ? 0 : -1}
                >
                    Scroll to Top
                </Button>
            </Pane>
        </Pane>
    );
};

export default ScrollToTopButton;
