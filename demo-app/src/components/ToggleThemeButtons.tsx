import { Notie } from "notie-markdown";
import { NotieThemes } from "notie-markdown";

const ToggleThemeButtons = ({
    setTheme,
}: {
    setTheme: (theme: NotieThemes) => void;
}) => {
    const themes: NotieThemes[] = [
        "default",
        "default dark",
        "Starlit Eclipse",
        "Starlit Eclipse Light",
    ];

    const themeButtons = themes.map((theme) => (
        <Button key={theme} onClick={() => setTheme(theme)}>
            {theme}
        </Button>
    ));

    return <Pane display="flex">{themeButtons}</Pane>;
};
