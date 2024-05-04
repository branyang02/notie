import { Pane } from 'evergreen-ui';
import React, { useEffect, useRef } from 'react';

import tikzjaxJs from '../utils/tikzjax?raw';

type TikZProps = {
  tikzScript: string;
};

const TikZ: React.FC<TikZProps> = ({ tikzScript }) => {
  const scriptRef = useRef<HTMLScriptElement>(null);

  useEffect(() => {
    if (scriptRef.current) {
      scriptRef.current.textContent = tikzScript;
      (window as Window & { TikZJax?: (element: HTMLScriptElement) => void })?.TikZJax?.(
        scriptRef.current,
      );
    }
  }, [tikzScript]);

  console.log(tikzjaxJs);

  return (
    <Pane
      className="tikz-drawing"
      display="flex"
      justifyContent="center"
      alignItems="center"
      flexGrow={1}
    >
      <script ref={scriptRef} type="text/tikz"></script>
    </Pane>
  );
};

export default TikZ;
