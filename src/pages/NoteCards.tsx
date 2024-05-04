import { Card, Heading, majorScale, Pane, Text } from 'evergreen-ui';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface NotesMetaData {
  title?: string;
  subtitle?: string;
  date?: string;
  link: string;
}

function NoteCards() {
  const [notesMetaData, setNotesMetaData] = useState<NotesMetaData[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchNotes = async () => {
      const markdownFiles = import.meta.glob('../../notes/*.md');
      const dateFilter = /\b(Spring|Summer|Fall|Autumn|Winter)\s+\d{4}\b/;

      const notesData = await Promise.all(
        Object.keys(markdownFiles).map(async (path) => {
          const rawFilePath = `${path}?raw`;
          const module = await import(/* @vite-ignore */ rawFilePath);
          const rawMDString = module.default;

          // Extracting title
          const title = /^#\s(.+)$/m.exec(rawMDString)?.[1].replace(/\*/g, '');
          // Extracting subtitle
          const fileName = path.split('/').pop()?.replace(/\.md$/, '');
          const subtitle = fileName?.replace(/-/g, ' ').replace(/.md$/, '');

          // Extracting date
          const date = dateFilter.exec(rawMDString)?.[0];

          return {
            title: title,
            subtitle: subtitle,
            link: `/notes/${fileName}`,
            date: date,
          };
        }),
      );

      notesData.sort((b, a) => sortNotesByDate(a.date, b.date));

      setNotesMetaData(notesData);
    };

    fetchNotes();
  }, []);

  const handleCardClick = (path: string) => {
    navigate(path);
  };

  function sortNotesByDate(dateA: string | undefined, dateB: string | undefined): number {
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
      const [season, year] = date.split(' ');

      const month = months[season as keyof SeasonMonthMap];
      if (!month) throw new Error(`Invalid season: ${season}`);

      return new Date(`${year}-${month}-01`);
    };

    const date1 = parseDate(dateA);
    const date2 = parseDate(dateB);

    return date1.getTime() - date2.getTime();
  }

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
          <Pane display="flex" justifyContent="space-between" alignItems="center">
            <Text size={300} color={'A7B6C2'}>
              {post.subtitle}
            </Text>
            {post.date && (
              <Text size={300} fontStyle="italic" color={'A7B6C2'}>
                {post.date}
              </Text>
            )}
          </Pane>
        </Card>
      ))}
    </Pane>
  );
}

export default NoteCards;
