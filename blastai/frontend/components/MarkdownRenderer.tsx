import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkEmoji from 'remark-emoji';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import mermaid from 'mermaid';
import 'katex/dist/katex.min.css';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

// Initialize Mermaid
if (typeof window !== 'undefined') {
  mermaid.initialize({
    startOnLoad: true,
    theme: 'dark',
    securityLevel: 'strict',
    themeVariables: {
      primaryColor: '#ffe067',
      primaryTextColor: '#1f1f1f',
      primaryBorderColor: '#ffd040',
      lineColor: '#ffe067',
      secondaryColor: '#2a2a2a',
      tertiaryColor: '#262626',
      background: '#1f1f1f',
      mainBkg: '#262626',
      secondBkg: '#2a2a2a',
      textColor: '#e5e5e5',
      border1: '#444',
      border2: '#555',
    }
  });
}

// Custom sanitization schema to allow safe HTML elements while preventing XSS
const sanitizeSchema = {
  ...defaultSchema,
  tagNames: [
    ...(defaultSchema.tagNames || []),
    // Add safe HTML elements that we use in the component
    'u', 'mark', 'kbd', 'details', 'summary', 'abbr',
    'dl', 'dt', 'dd', 'figure', 'figcaption', 'small',
    'section', 'article', 'aside', 'sup', 'sub',
  ],
  attributes: {
    ...defaultSchema.attributes,
    // Allow specific safe attributes
    '*': ['className', 'id', 'style'],
    'abbr': ['title'],
    'input': ['type', 'checked', 'disabled'],
    'details': ['open'],
  },
  // Explicitly strip dangerous protocols
  protocols: {
    ...defaultSchema.protocols,
    href: ['http', 'https', 'mailto'],
    src: ['http', 'https'],
  },
};

// Mermaid diagram component
const MermaidDiagram = ({ chart }: { chart: string }) => {
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const renderDiagram = async () => {
      try {
        const id = `mermaid-${Math.random().toString(36).slice(2, 11)}`;
        const { svg: renderedSvg } = await mermaid.render(id, chart);
        setSvg(renderedSvg);
      } catch (err: any) {
        setError(err.message || 'Failed to render diagram');
        console.error('Mermaid rendering error:', err);
      }
    };

    if (chart && typeof window !== 'undefined') {
      renderDiagram();
    }
  }, [chart]);

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4 mb-4 text-red-300 text-sm">
        <strong>Diagram Error:</strong> {error}
      </div>
    );
  }

  return (
    <div
      className="mermaid-diagram my-4 flex justify-center bg-[#262626] rounded-lg p-4 overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
};

const extractText = (children: any): string => {
  if (typeof children === 'string') {
    return children;
  }
  if (Array.isArray(children)) {
    return children.map(extractText).join('');
  }
  if (children?.props?.children) {
    return extractText(children.props.children);
  }
  return '';
};

// Code block with copy button and language badge
const CodeBlock = ({ inline, className, children, ...props }: any) => {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';
  const codeString = extractText(children).replace(/\n$/, '');

  const handleCopy = async () => {
    try {
      // Try modern clipboard API first
      await navigator.clipboard.writeText(codeString);
      setCopied(true);
      setCopyError(false);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      // Fallback to older execCommand method
      try {
        const textArea = document.createElement('textarea');
        textArea.value = codeString;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        const successful = document.execCommand('copy');
        document.body.removeChild(textArea);
        
        if (successful) {
          setCopied(true);
          setCopyError(false);
          setTimeout(() => setCopied(false), 2000);
        } else {
          throw new Error('execCommand failed');
        }
      } catch (fallbackErr) {
        // Both methods failed
        console.error('Failed to copy text:', err, fallbackErr);
        setCopyError(true);
        setTimeout(() => setCopyError(false), 2000);
      }
    }
  };

  if (inline) {
    return (
      <code
        className="bg-[#2a2a2a] text-[#ffe067] px-1.5 py-0.5 rounded text-sm font-mono"
        {...props}
      >
        {children}
      </code>
    );
  }

  // Check if this is a mermaid diagram
  if (language === 'mermaid') {
    return <MermaidDiagram chart={codeString} />;
  }

  return (
    <div className="relative group mb-4">
      <button
        onClick={handleCopy}
        className={`absolute top-2 right-2 p-1.5 rounded-md hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-10 ${
          copyError ? 'opacity-100' : ''
        }`}
        aria-label={copyError ? 'Failed to copy code' : copied ? 'Code copied!' : 'Copy code'}
        title={copyError ? 'Failed to copy code' : copied ? 'Copied!' : 'Copy to clipboard'}
      >
        <svg
          width="18"
          height="18"
          viewBox="0 0 20 20"
          fill="none"
          className={copyError ? 'text-red-400' : copied ? 'text-green-400' : 'text-gray-400'}
        >
          {copyError ? (
            // Error X icon
            <path
              d="M10 18C14.4183 18 18 14.4183 18 10C18 5.58172 14.4183 2 10 2C5.58172 2 2 5.58172 2 10C2 14.4183 5.58172 18 10 18ZM7.70711 6.29289C7.31658 5.90237 6.68342 5.90237 6.29289 6.29289C5.90237 6.68342 5.90237 7.31658 6.29289 7.70711L8.58579 10L6.29289 12.2929C5.90237 12.6834 5.90237 13.3166 6.29289 13.7071C6.68342 14.0976 7.31658 14.0976 7.70711 13.7071L10 11.4142L12.2929 13.7071C12.6834 14.0976 13.3166 14.0976 13.7071 13.7071C14.0976 13.3166 14.0976 12.6834 13.7071 12.2929L11.4142 10L13.7071 7.70711C14.0976 7.31658 14.0976 6.68342 13.7071 6.29289C13.3166 5.90237 12.6834 5.90237 12.2929 6.29289L10 8.58579L7.70711 6.29289Z"
              fill="currentColor"
            />
          ) : copied ? (
            // Success checkmark icon
            <path
              d="M15.1883 5.10908C15.3699 4.96398 15.6346 4.96153 15.8202 5.11592C16.0056 5.27067 16.0504 5.53125 15.9403 5.73605L15.8836 5.82003L8.38354 14.8202C8.29361 14.9279 8.16242 14.9925 8.02221 14.9989C7.88203 15.0051 7.74545 14.9526 7.64622 14.8534L4.14617 11.3533L4.08172 11.2752C3.95384 11.0811 3.97542 10.817 4.14617 10.6463C4.31693 10.4755 4.58105 10.4539 4.77509 10.5818L4.85321 10.6463L7.96556 13.7586L15.1161 5.1794L15.1883 5.10908Z"
              fill="currentColor"
            />
          ) : (
            // Copy icon
            <>
              <rect x="8" y="8" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
              <path d="M5 12V5C5 3.89543 5.89543 3 7 3H12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </>
          )}
        </svg>
      </button>

      <div className="rounded-lg overflow-hidden border border-[#2a2a2a] bg-[#1a1a1a]">
        {language && (
          <div className="px-4 pt-3 pb-1">
            <span className="text-[12px] text-gray-500 font-mono tracking-wide">
              {language}
            </span>
          </div>
        )}
        <pre className="bg-[#1a1a1a] overflow-x-auto !mt-0 !mb-0">
          <code
            className={`block px-4 ${language ? 'pt-1 pb-4' : 'py-4'} text-[13px] leading-relaxed !bg-transparent ${className || ''}`}
            style={{ background: 'transparent' }}
            {...props}
          >
            {children}
          </code>
        </pre>
      </div>
    </div>
  );
};

// Custom precomponent to detect mermaid diagrams
const PreComponent = ({ children, ...props }: any) => {
  const childProps = children?.props;
  const className = childProps?.className || '';
  const match = /language-mermaid/.exec(className);

  if (match) {
    const code = String(childProps?.children || '').replace(/\n$/, '');
    return <MermaidDiagram chart={code} />;
  }

  return <pre {...props}>{children}</pre>;
};

export const MarkdownRenderer = ({ content, className = '' }: MarkdownRendererProps) => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Don't render anything on the server to avoid hydration mismatch
  if (!isClient) {
    return <div className={`markdown-content ${className}`} />;
  }

  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkEmoji]}
        rehypePlugins={[
          rehypeRaw,
          [rehypeSanitize, sanitizeSchema],
          rehypeKatex,
          rehypeHighlight,
        ]}
        components={{
          // Headings
          h1: ({ node, ...props }) => (
            <h1 className="text-3xl font-bold mb-4 mt-6 text-gray-100 border-b border-gray-700 pb-2" {...props} />
          ),
          h2: ({ node, ...props }) => (
            <h2 className="text-2xl font-bold mb-3 mt-5 text-gray-100 border-b border-gray-800 pb-2" {...props} />
          ),
          h3: ({ node, ...props }) => (
            <h3 className="text-xl font-bold mb-2 mt-4 text-gray-100" {...props} />
          ),
          h4: ({ node, ...props }) => (
            <h4 className="text-lg font-semibold mb-2 mt-3 text-gray-200" {...props} />
          ),
          h5: ({ node, ...props }) => (
            <h5 className="text-base font-semibold mb-2 mt-3 text-gray-200" {...props} />
          ),
          h6: ({ node, ...props }) => (
            <h6 className="text-sm font-semibold mb-2 mt-3 text-gray-300" {...props} />
          ),

          // Paragraphs
          p: ({ node, ...props }) => (
            <p className="mb-4 text-gray-300 leading-relaxed" {...props} />
          ),

          // Lists
          ul: ({ node, ...props }) => (
            <ul className="list-disc list-inside mb-4 space-y-2 text-gray-300 ml-4" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="list-decimal list-inside mb-4 space-y-2 text-gray-300 ml-4" {...props} />
          ),
          li: ({ node, children, ...props }) => {
            // Check if this is a task list item
            const hasCheckbox = typeof children === 'object' &&
              Array.isArray(children) &&
              children.some((child: any) => child?.type === 'input');

            return (
              <li className={`text-gray-300 ${hasCheckbox ? 'list-none -ml-4' : ''}`} {...props}>
                {children}
              </li>
            );
          },

          // Code blocks with copy button and language badge
          code: CodeBlock,
          pre: PreComponent,

          // Blockquotes
          blockquote: ({ node, ...props }) => (
            <blockquote
              className="border-l-4 border-[#ffe067] pl-4 py-2 mb-4 text-gray-400 italic bg-[#262626] rounded-r"
              {...props}
            />
          ),

          // Links
          a: ({ node, ...props }) => (
            <a
              className="text-[#ffe067] hover:text-[#ffd040] underline transition-colors break-words"
              target="_blank"
              rel="noopener noreferrer"
              {...props}
            />
          ),

          // Horizontal rule
          hr: ({ node, ...props }) => (
            <hr className="border-gray-700 my-8" {...props} />
          ),

          // Tables
          table: ({ node, ...props }) => (
            <div className="overflow-x-auto mb-4 rounded-lg border border-gray-700">
              <table className="min-w-full divide-y divide-gray-700" {...props} />
            </div>
          ),
          thead: ({ node, ...props }) => (
            <thead className="bg-[#262626]" {...props} />
          ),
          tbody: ({ node, ...props }) => (
            <tbody className="bg-[#1f1f1f] divide-y divide-gray-700" {...props} />
          ),
          tr: ({ node, ...props }) => (
            <tr className="hover:bg-[#2a2a2a] transition-colors" {...props} />
          ),
          th: ({ node, ...props }) => (
            <th
              className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider"
              {...props}
            />
          ),
          td: ({ node, ...props }) => (
            <td className="px-6 py-4 text-sm text-gray-400 whitespace-nowrap" {...props} />
          ),

          // Strong (bold)
          strong: ({ node, ...props }) => (
            <strong className="font-bold text-gray-100" {...props} />
          ),

          // Emphasis (italic)
          em: ({ node, ...props }) => (
            <em className="italic text-gray-200" {...props} />
          ),

          // Delete (strikethrough)
          del: ({ node, ...props }) => (
            <del className="line-through text-gray-500" {...props} />
          ),

          // Images
          img: ({ node, ...props }) => (
            <img
              className="rounded-lg max-w-full h-auto my-4 border border-gray-700"
              {...props}
              alt={props.alt || ''}
            />
          ),

          // Superscript and subscript
          sup: ({ node, ...props }) => (
            <sup className="text-xs align-super" {...props} />
          ),
          sub: ({ node, ...props }) => (
            <sub className="text-xs align-sub" {...props} />
          ),

          // Underline (from HTML)
          u: ({ node, ...props }) => (
            <u className="underline decoration-gray-400" {...props} />
          ),

          // Mark/highlight (from HTML)
          mark: ({ node, ...props }) => (
            <mark className="bg-[#ffe067] text-black px-1 py-0.5 rounded" {...props} />
          ),

          // Keyboard
          kbd: ({ node, ...props }) => (
            <kbd className="bg-[#2a2a2a] border border-gray-600 text-gray-300 px-2 py-1 rounded text-sm font-mono shadow-sm" {...props} />
          ),

          // Details/Summary (collapsible)
          details: ({ node, ...props }) => (
            <details className="bg-[#262626] rounded-lg p-4 mb-4 border border-gray-700 hover:border-gray-600 transition-colors" {...props} />
          ),
          summary: ({ node, ...props }) => (
            <summary className="cursor-pointer font-semibold text-gray-200 hover:text-[#ffe067] transition-colors select-none" {...props} />
          ),

          // Abbreviation
          abbr: ({ node, ...props }) => (
            <abbr className="border-b border-dotted border-gray-500 cursor-help" {...props} />
          ),

          // Definition list
          dl: ({ node, ...props }) => (
            <dl className="mb-4 space-y-2" {...props} />
          ),
          dt: ({ node, ...props }) => (
            <dt className="font-bold text-gray-200 mt-2" {...props} />
          ),
          dd: ({ node, ...props }) => (
            <dd className="ml-6 text-gray-400 mb-2 pl-4 border-l-2 border-gray-700" {...props} />
          ),

          // Figure and figcaption
          figure: ({ node, ...props }) => (
            <figure className="my-4" {...props} />
          ),
          figcaption: ({ node, ...props }) => (
            <figcaption className="text-sm text-gray-500 text-center mt-2 italic" {...props} />
          ),

          // Break
          br: ({ node, ...props }) => (
            <br {...props} />
          ),

          // Span (for custom styling)
          span: ({ node, ...props }) => (
            <span {...props} />
          ),

          // Div (for custom containers)
          div: ({ node, ...props }) => (
            <div {...props} />
          ),

          // Small text
          small: ({ node, ...props }) => (
            <small className="text-xs text-gray-500" {...props} />
          ),

          // Section
          section: ({ node, ...props }) => (
            <section className="my-4" {...props} />
          ),

          // Article
          article: ({ node, ...props }) => (
            <article className="my-4" {...props} />
          ),

          // Aside
          aside: ({ node, ...props }) => (
            <aside className="border-l-4 border-gray-600 pl-4 my-4 text-gray-400 bg-[#262626] py-2 rounded-r" {...props} />
          ),

          // Input (for checkboxes in task lists)
          input: ({ node, ...props }) => (
            <input
              className="mr-2 accent-[#ffe067] cursor-pointer"
              {...props}
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
