import { Pane, Button, ClipboardIcon, TickCircleIcon } from "evergreen-ui";
import React, { useState } from "react";

const CodeHeader = ({
    language,
    code,
    copyButtonHoverColor,
}: {
    language: string;
    code: string;
    copyButtonHoverColor?: string;
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

    const buttonStyle: React.CSSProperties = {
        backgroundColor: isHovered ? copyButtonHoverColor : "transparent",
        transition: "background-color 0.3s ease",
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
            className="code-header"
        >
            <span>{language}</span>
            <Button
                iconBefore={isCopied ? <TickCircleIcon /> : <ClipboardIcon />}
                appearance="minimal"
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
                onClick={isCopied ? resetCopy : copyToClipboard}
                height={24}
                paddingY={1}
                paddingX={8}
                style={buttonStyle}
                color="inherit"
            >
                {isCopied ? "Copied!" : "Copy Code"}
            </Button>
        </Pane>
    );
};

export default CodeHeader;
