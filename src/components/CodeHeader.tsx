import { Pane, Button, ClipboardIcon, TickCircleIcon } from "evergreen-ui";
import { useEffect, useRef, useState } from "react";
import styles from "../styles/Notie.module.css";

const CodeHeader = ({ language, code }: { language: string; code: string }) => {
    const [isCopied, setIsCopied] = useState(false);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const isMountedRef = useRef(true);

    useEffect(() => {
        isMountedRef.current = true;
        return () => {
            isMountedRef.current = false;
            if (timerRef.current) {
                clearTimeout(timerRef.current);
                timerRef.current = null;
            }
        };
    }, []);

    const copyToClipboard = () => {
        if (!navigator.clipboard?.writeText) {
            return;
        }
        navigator.clipboard
            .writeText(code)
            .then(() => {
                if (!isMountedRef.current) {
                    return;
                }
                setIsCopied(true);
                if (timerRef.current) {
                    clearTimeout(timerRef.current);
                }
                timerRef.current = setTimeout(() => {
                    setIsCopied(false);
                    timerRef.current = null;
                }, 2000);
            })
            .catch(() => {
                // Clipboard write failed (e.g. permission denied or insecure
                // origin); leave the button in its default state.
            });
    };

    const resetCopy = () => {
        setIsCopied(false);
        if (timerRef.current) {
            clearTimeout(timerRef.current);
            timerRef.current = null;
        }
    };

    return (
        <Pane
            paddingY={1}
            paddingX={8}
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            style={{
                fontSize: "0.8rem",
                borderRadius: "10px 10px 0 0",
            }}
            className={styles["code-header"]}
        >
            <span>{language}</span>
            <Button
                iconBefore={isCopied ? <TickCircleIcon /> : <ClipboardIcon />}
                appearance="minimal"
                onClick={isCopied ? resetCopy : copyToClipboard}
                height={24}
                paddingY={1}
                paddingX={8}
                color="inherit"
                className={styles["copy-button"]}
            >
                {isCopied ? "Copied!" : "Copy Code"}
            </Button>
        </Pane>
    );
};

export default CodeHeader;
