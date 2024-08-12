import {
    Pane,
    Button,
    ClipboardIcon,
    TickCircleIcon,
    Text,
} from "evergreen-ui";
import React, { useState } from "react";

const CodeHeader = ({
    language,
    code,
    darkMode,
}: {
    language: string;
    code: string;
    darkMode: boolean;
}) => {
    const [isCopied, setIsCopied] = useState(false);
    const [isHovered, setIsHovered] = useState(false);
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

    // Button Dark Mode Style
    const buttonStyle: React.CSSProperties = {
        backgroundColor: darkMode && isHovered ? "#444" : "transparent",
        transition: "background-color 0.3s ease",
    };

    const defaultStyle: React.CSSProperties = {
        WebkitFontSmoothing: "antialiased",
        appearance: "none",
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
                backgroundColor: "#afb8c133",
            }}
        >
            <span>{language}</span>
            <Button
                iconBefore={
                    isCopied ? (
                        <TickCircleIcon
                            color={darkMode ? "muted" : "default"}
                        />
                    ) : (
                        <ClipboardIcon color={darkMode ? "muted" : "default"} />
                    )
                }
                appearance="minimal"
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
                onClick={isCopied ? resetCopy : copyToClipboard}
                height={24}
                paddingY={1}
                paddingX={8}
                style={darkMode ? buttonStyle : defaultStyle}
            >
                <Text color={darkMode ? "tint2" : "default"} size={300}>
                    {isCopied ? "Copied!" : "Copy Code"}
                </Text>
            </Button>
        </Pane>
    );
};

export default CodeHeader;
