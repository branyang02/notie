import "katex/dist/katex.min.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "../styles/notie.css";

import React, { useRef, useMemo, useState, useEffect } from "react";
import { Pane } from "evergreen-ui";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import CodeBlock from "./CodeBlock";
import TikZ from "./TikZ";
import StaticCodeBlock from "./StaticCodeBlock";
import ScrollToTopButton from "./ScrollToTopButton";
import NoteToc from "./NoteToc";

export interface NotieProps {
  markdown: string;
  darkMode?: boolean;
  style?: React.CSSProperties;
}

type CodeProps = React.HTMLAttributes<HTMLElement> & {
  node?: unknown;
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
};

function processMarkdown(markdownContent: string): string {
  const pattern = /```(\w+)/g;
  const processedContent = markdownContent.replace(pattern, "```language-$1");

  return processedContent;
}

const MarkdownRenderer: React.FC<{
  markdownContent: string;
  darkMode?: boolean;
}> = React.memo(({ markdownContent, darkMode }) => {
  const components = useMemo(
    () => ({
      code({ inline, className, children, ...props }: CodeProps) {
        const match = /\w+/.exec(className || "");

        if (!inline && match) {
          const language = className?.split("language-").pop() || "";
          const content = Array.isArray(children)
            ? children.join("")
            : children;
          const code = String(content).replace(/\n$/, "");
          if (language.includes("execute-")) {
            return (
              <CodeBlock
                initialCode={code}
                language={language.split("-").pop()}
                darkMode={darkMode}
              />
            );
          }
          if (language === "tikz") {
            return <TikZ tikzScript={code} />;
          }
          return (
            <StaticCodeBlock
              code={code}
              language={language}
              darkMode={darkMode}
            />
          );
        } else {
          return (
            <code className={className} {...props}>
              {children}
            </code>
          );
        }
      },
    }),
    [darkMode]
  );

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex, rehypeRaw, rehypeHighlight, rehypeSlug]}
      components={components}
    >
      {markdownContent}
    </ReactMarkdown>
  );
});

const Notie: React.FC<NotieProps> = ({ markdown, darkMode, style }) => {
  const markdownContent = processMarkdown(markdown);
  const contentRef = useRef<HTMLDivElement>(null);
  const [activeId, setActiveId] = useState<string>("");

  // Effect to observe headings and update activeId
  useEffect(() => {
    const observerOptions = {
      rootMargin: "0px 0px -80% 0px",
      threshold: 1.0,
    };

    if (!contentRef.current) return;

    const headings = contentRef.current.querySelectorAll(
      "h1, h2, h3, h4, h5, h6"
    );
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setActiveId(entry.target.id);
        }
      });
    }, observerOptions);

    headings.forEach((heading) => observer.observe(heading));

    return () => {
      headings.forEach((heading) => observer.unobserve(heading));
    };
  }, [markdownContent]);

  return (
    <Pane background={darkMode ? "#333" : "white"} style={style}>
      <Pane className="mw-page-container-inner">
        <Pane className="vector-column-start">
          <NoteToc
            markdownContent={markdownContent}
            darkMode={darkMode}
            activeId={activeId}
          />
        </Pane>
        <Pane className="mw-content-container">
          <Pane
            className={`blog-content ${darkMode ? "dark-mode" : ""}`}
            ref={contentRef}
          >
            <MarkdownRenderer
              markdownContent={markdownContent}
              darkMode={darkMode}
            />
            <ScrollToTopButton />
          </Pane>
        </Pane>
      </Pane>
    </Pane>
  );
};

export default Notie;
