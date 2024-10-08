import { Pane, Button, ClipboardIcon, TickCircleIcon } from "evergreen-ui";
import { useState } from "react";
import styles from "../styles/Notie.module.css";

const CodeHeader = ({ language, code }: { language: string; code: string }) => {
    const [isCopied, setIsCopied] = useState(false);
    const [timer, setTimer] = useState<NodeJS.Timeout | null>(null);

    const copyToClipboard = () => {
        navigator.clipboard.writeText(code).then(() => {
            setIsCopied(true);
            if (timer) {
                clearTimeout(timer);
            }
            const newTimer = setTimeout(() => {
                setIsCopied(false);
            }, 2000);
            setTimer(newTimer);
        });
    };

    const resetCopy = () => {
        setIsCopied(false);
        if (timer) {
            clearTimeout(timer);
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
