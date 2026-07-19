import { Heading, majorScale, Pane, Paragraph } from "evergreen-ui";
import { Link } from "react-router-dom";
import { useTheme } from "../context/useTheme";

const NotFound = () => {
    const { theme } = useTheme();
    const darkMode = theme === "default dark" || theme === "Starlit Eclipse";

    return (
        <Pane
            display="flex"
            flexDirection="column"
            alignItems="center"
            padding={majorScale(4)}
            style={{ margin: "0 auto" }}
        >
            <Heading
                size={800}
                marginBottom={majorScale(2)}
                color={darkMode ? "white" : "default"}
            >
                Page Not Found
            </Heading>
            <Paragraph color={darkMode ? "white" : "default"}>
                The page you are looking for does not exist.{" "}
                <Link
                    to="/"
                    style={{ color: darkMode ? "white" : "#3366FF" }}
                >
                    Go back home
                </Link>
            </Paragraph>
        </Pane>
    );
};

export default NotFound;
