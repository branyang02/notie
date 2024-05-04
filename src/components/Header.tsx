import { Avatar, Heading, majorScale, Pane } from 'evergreen-ui';
import { useNavigate } from 'react-router-dom';

import { useDarkMode } from '../context/DarkModeContext';

const Header = () => {
  const { darkMode } = useDarkMode();
  const navigate = useNavigate();

  return (
    <Pane display="flex" borderBottom="default" width="100%" elevation={2}>
      <Pane padding={20} paddingLeft={majorScale(5)} display="flex">
        <Heading
          size={600}
          color={darkMode ? 'white' : 'black'}
          // go to home on click
          cursor="pointer"
          onClick={() => navigate('/')}
        >
          NOTIE
        </Heading>
      </Pane>

      <Pane
        flex={1}
        display="flex"
        justifyContent="flex-end"
        alignItems="center"
        paddingRight={majorScale(5)}
      >
        <Avatar name="John Doe" shape="square" size={40} marginRight={16} />
      </Pane>
    </Pane>
  );
};

export default Header;
