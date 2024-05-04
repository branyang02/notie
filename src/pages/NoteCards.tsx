import { Card, Heading, majorScale, Pane, Text } from 'evergreen-ui';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface NotesMetaData {
  title?: string;
  subtitle?: string;
  link: string;
}

function NoteCards() {
  const [notesMetaData, setNotesMetaData] = useState<NotesMetaData[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchNotes = async () => {
      const markdownFiles = import.meta.glob('../../notes/*.md');

      const notesData = await Promise.all(
        Object.keys(markdownFiles).map(async (path) => {
          const rawFilePath = `${path}?raw`;
          const module = await import(/* @vite-ignore */ rawFilePath);
          const rawMDString = module.default;

          const title = /^#\s(.+)$/m.exec(rawMDString)?.[1].replace(/\*/g, '');

          const fileName = path.split('/').pop()?.replace(/\.md$/, '');
          const subtitle = fileName?.replace(/-/g, ' ').replace(/.md$/, '');
          return {
            title: title,
            subtitle: subtitle,
            link: `/notes/${fileName}`,
          };
        }),
      );

      setNotesMetaData(notesData);
    };

    fetchNotes();
  }, []);

  const handleCardClick = (path: string) => {
    navigate(path);
  };

  return (
    <Pane padding={majorScale(2)}>
      {notesMetaData.map((post, index) => (
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
        >
          <Heading size={500} marginBottom={majorScale(1)} className="note-postHeading">
            {post.title}
          </Heading>
          <Text size={300} color={'A7B6C2'}>
            {post.subtitle}
          </Text>
        </Card>
      ))}
    </Pane>
  );
}

export default NoteCards;
