import { Card, Heading, majorScale, Pane, Spinner, Text } from "evergreen-ui";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../context/useTheme";

const modules = import.meta.glob("../assets/**.md", {
    query: "?raw",
    import: "default",
});

interface NotesMetaData {
    title?: string;
    subtitle?: string;
    date?: string;
    link: string;
}

const Examples = () => {
    const [notesMetaData, setNotesMetaData] = useState<NotesMetaData[]>([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();
    const { theme } = useTheme();
    const darkMode = theme === "default dark" || theme === "Starlit Eclipse";

    useEffect(() => {
        async function fetchNotes() {
            const notesData: NotesMetaData[] = [];
            for (const path in modules) {
                const markdown = await modules[path]();
                const rawMDString = markdown as string;
                const title = /^#\s(.+)$/m
                    .exec(rawMDString)?.[1]
                    .replace(/\*/g, "");
                const fileName = path.split("/").pop()?.replace(/\.md$/, "");
                const subtitle = fileName
                    ?.replace(/-/g, " ")
                    .replace(/.md$/, "");
                const dateFilter =
                    /\b(Spring|Summer|Fall|Autumn|Winter)\s+\d{4}\b/;
                const date = dateFilter.exec(rawMDString)?.[0];

                notesData.push({
                    title: title,
                    subtitle: subtitle,
                    link: `/examples/${fileName}`,
                    date: date,
                });
            }
            notesData.sort((b, a) => sortNotesByDate(a.date, b.date));
            setNotesMetaData(notesData);
            setLoading(false);
        }

        function sortNotesByDate(
            dateA: string | undefined,
            dateB: string | undefined,
        ): number {
            if (!dateA || !dateB) return 0;

            type SeasonMonthMap = {
                Spring: number;
                Summer: number;
                Fall: number;
                Autumn: number;
                Winter: number;
            };

            const months: SeasonMonthMap = {
                Spring: 1,
                Summer: 4,
                Fall: 9,
                Autumn: 9,
                Winter: 12,
            };

            const parseDate = (date: string): Date => {
                const [season, year] = date.split(" ");

                const month = months[season as keyof SeasonMonthMap];
                if (!month) throw new Error(`Invalid season: ${season}`);

                return new Date(`${year}-${month}-01`);
            };

            const date1 = parseDate(dateA);
            const date2 = parseDate(dateB);

            return date1.getTime() - date2.getTime();
        }

        fetchNotes();
    }, []);

    const handleCardClick = (path: string) => {
        navigate(path);
    };

    if (loading) {
        return (
            <Pane
                display="flex"
                alignItems="center"
                justifyContent="center"
                style={{
                    margin: "0 auto",
                }}
            >
                <Spinner />
            </Pane>
        );
    }

    const exampleNotes = notesMetaData.filter((note) =>
        note.title?.includes("Example"),
    );
    const otherNotes = notesMetaData.filter(
        (note) => !note.title?.includes("Example"),
    );

    return (
        <Pane
            padding={majorScale(2)}
            style={{
                margin: "0 auto",
            }}
            display="flex"
            flexDirection="row"
            justifyContent="space-between"
        >
            <Pane width="48%" display="flex" flexDirection="column">
                <Heading
                    size={600}
                    marginBottom={majorScale(2)}
                    color={darkMode ? "white" : "default"}
                >
                    Examples
                </Heading>
                {exampleNotes.map((post, index) => (
                    <Card
                        key={index}
                        className="BlogCard"
                        elevation={1}
                        hoverElevation={2}
                        marginY={majorScale(1)}
                        padding={majorScale(2)}
                        display="flex"
                        flexDirection="column"
                        justifyContent="space-between"
                        onClick={() => handleCardClick(post.link)}
                        cursor="pointer"
                        background={darkMode ? "overlay" : "tint2"}
                        color={darkMode ? "white" : "default"}
                    >
                        <Heading
                            size={500}
                            marginBottom={majorScale(1)}
                            className="note-postHeading"
                            color={darkMode ? "white" : "default"}
                        >
                            {post.title}
                        </Heading>
                        <Pane
                            display="flex"
                            justifyContent="space-between"
                            alignItems="center"
                        >
                            <Text size={300} color={"A7B6C2"}>
                                {post.subtitle}
                            </Text>
                        </Pane>
                    </Card>
                ))}
            </Pane>

            <Pane width="48%" display="flex" flexDirection="column">
                <Heading
                    size={600}
                    marginBottom={majorScale(2)}
                    color={darkMode ? "white" : "default"}
                >
                    Full Demos
                </Heading>
                {otherNotes.map((post, index) => (
                    <Card
                        key={index}
                        className="BlogCard"
                        elevation={1}
                        hoverElevation={2}
                        marginY={majorScale(1)}
                        padding={majorScale(2)}
                        display="flex"
                        flexDirection="column"
                        justifyContent="space-between"
                        onClick={() => handleCardClick(post.link)}
                        cursor="pointer"
                        background={darkMode ? "overlay" : "tint2"}
                        color={darkMode ? "white" : "default"}
                    >
                        <Heading
                            size={500}
                            marginBottom={majorScale(1)}
                            className="note-postHeading"
                            color={darkMode ? "white" : "default"}
                        >
                            {post.title}
                        </Heading>
                        <Pane
                            display="flex"
                            justifyContent="space-between"
                            alignItems="center"
                        >
                            <Text size={300} color={"A7B6C2"}>
                                {post.subtitle}
                            </Text>
                        </Pane>
                    </Card>
                ))}
            </Pane>
        </Pane>
    );
};

export default Examples;
