import { MarkdownRenderer } from '../components/MarkdownRenderer';

const markdownExamples = `# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6

---

## Text Formatting

This is **bold text** and this is *italic text*. You can also use ***bold and italic*** together.

Here's some ~~strikethrough text~~ and some \`inline code\`.

This is <u>underlined text</u> and this is <mark>highlighted text</mark>.

---

## Lists

### Unordered List
- Item 1
- Item 2
  - Nested item 2.1
  - Nested item 2.2
- Item 3

### Ordered List
1. First item
2. Second item
3. Third item
   1. Nested item
   2. Another nested item

### Task List
- [x] Completed task
- [ ] Incomplete task
- [x] Another completed task
- [ ] Another incomplete task

---

## Blockquotes

> This is a blockquote.
> It can span multiple lines.
> 
> And can have multiple paragraphs.

---

## Links and Images

[This is a link to Google](https://www.google.com)

[This is a link with title](https://www.example.com "Example Website")

![Image alt text](https://via.placeholder.com/600x300)

---

## Code Blocks

### Python Code
\`\`\`python
def hello_world():
    """A simple hello world function"""
    print("Hello, World!")
    return True

class MyClass:
    def __init__(self, name):
        self.name = name
\`\`\`

### JavaScript Code
\`\`\`javascript
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

const result = fibonacci(10);
console.log(result);
\`\`\`

### TypeScript Code
\`\`\`typescript
interface User {
  id: number;
  name: string;
  email: string;
}

const user: User = {
  id: 1,
  name: "John Doe",
  email: "john@example.com"
};
\`\`\`

### Bash/Shell Code
\`\`\`bash
#!/bin/bash
echo "Hello, World!"
npm install
npm run dev
\`\`\`

---

## Tables

| Feature | Status | Priority |
|---------|--------|----------|
| Headers | ✅ Done | High |
| Lists | ✅ Done | High |
| Tables | ✅ Done | Medium |
| Mermaid | ✅ Done | Low |

| Left Aligned | Center Aligned | Right Aligned |
|:-------------|:--------------:|--------------:|
| Left | Center | Right |
| A | B | C |

---

## Math Equations (KaTeX)

Inline math: \\(E = mc^2\\)

Block math:

\\[
\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}
\\]

\\[
\\frac{d}{dx}\\left( \\int_{a}^{x} f(u)\\,du\\right)=f(x)
\\]

---

## Mermaid Diagrams

### Flowchart
\`\`\`mermaid
graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Debug]
    D --> B
    C --> E[End]
\`\`\`

### Sequence Diagram
\`\`\`mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Database
    
    User->>Frontend: Submit form
    Frontend->>Backend: POST /api/data
    Backend->>Database: INSERT query
    Database-->>Backend: Success
    Backend-->>Frontend: 200 OK
    Frontend-->>User: Show success message
\`\`\`

### Pie Chart
\`\`\`mermaid
pie title Browser Usage
    "Chrome" : 65
    "Firefox" : 15
    "Safari" : 12
    "Edge" : 8
\`\`\`

---

## HTML Elements

### Keyboard Input
Press <kbd>Ctrl</kbd> + <kbd>C</kbd> to copy
Press <kbd>Cmd</kbd> + <kbd>V</kbd> to paste

### Abbreviations
The <abbr title="HyperText Markup Language">HTML</abbr> specification is maintained by the <abbr title="World Wide Web Consortium">W3C</abbr>.

### Superscript and Subscript
E = mc<sup>2</sup>

H<sub>2</sub>O is water

### Small Text
<small>This is small text for fine print</small>

---

## Collapsible Sections (Details/Summary)

<details>
<summary>Click to expand</summary>

This content is hidden by default and can be revealed by clicking the summary.

You can include:
- Lists
- **Formatted text**
- \`code\`
- And more!

</details>

<details>
<summary>Another collapsible section</summary>

### Nested content

This can contain any markdown content, including code blocks:

\`\`\`python
print("Hidden code!")
\`\`\`

</details>

---

## Definition Lists

<dl>
  <dt>Term 1</dt>
  <dd>Definition of term 1</dd>
  
  <dt>Term 2</dt>
  <dd>Definition of term 2</dd>
  
  <dt>HTML</dt>
  <dd>HyperText Markup Language</dd>
  
  <dt>CSS</dt>
  <dd>Cascading Style Sheets</dd>
</dl>

---

## Figure and Caption

<figure>
  <img src="https://picsum.photos/200" alt="Placeholder" />
  <figcaption>Figure 1: This is a caption for the image above</figcaption>
</figure>

---

## Emojis

:rocket: :fire: :tada: :heart: :star: :sparkles: :zap: :boom: :+1: :smile:

---

## Complex Nested Content

1. **First Level**
   - Second level with *italic*
   - Second level with \`code\`
     - Third level with [link](https://example.com)
     - Third level with **bold**
       - Fourth level with ~~strikethrough~~
       
2. **Another First Level**
   > Blockquote inside list
   > 
   > With multiple lines
   
3. **Code in List**
   \`\`\`javascript
   console.log("Code block inside list");
   \`\`\`

---

## Mixed Content Example

Here's a paragraph with **bold**, *italic*, \`code\`, [link](https://example.com), and ~~strikethrough~~ all in one line.

You can also use <mark>highlighting</mark>, <u>underline</u>, and even <abbr title="Cascading Style Sheets">CSS</abbr> abbreviations together.

Math inline: \\(x^2 + y^2 = z^2\\) mixed with regular text.

---

## Long Code Block with Line Numbers

\`\`\`python
# A longer Python example
class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        """Add two numbers"""
        self.result = x + y
        return self.result
    
    def subtract(self, x, y):
        """Subtract y from x"""
        self.result = x - y
        return self.result
    
    def multiply(self, x, y):
        """Multiply two numbers"""
        self.result = x * y
        return self.result
    
    def divide(self, x, y):
        """Divide x by y"""
        if y == 0:
            raise ValueError("Cannot divide by zero")
        self.result = x / y
        return self.result

# Usage example
calc = Calculator()
print(calc.add(10, 5))      # 15
print(calc.subtract(10, 5))  # 5
print(calc.multiply(10, 5))  # 50
print(calc.divide(10, 5))    # 2.0
\`\`\`

---

## End of Examples

This test page includes examples of all major markdown features supported by the MarkdownRenderer component.
`;

export default function TestMarkdown() {
  return (
    <div className="min-h-screen bg-[#1f1f1f] py-12 px-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-[#ffe067] mb-2">
            Markdown Renderer Test Page
          </h1>
          <p className="text-gray-400">
            Testing all markdown features and components
          </p>
        </div>
        
        <div className="bg-[#262626] rounded-lg p-8 shadow-xl border border-gray-800">
          <MarkdownRenderer content={markdownExamples} />
        </div>
        
        <div className="mt-8 text-center">
          <a
            href="/"
            className="inline-block px-6 py-3 bg-[#ffe067] text-black rounded-lg font-semibold hover:bg-[#ffd040] transition-colors"
          >
            ← Back to Main App
          </a>
        </div>
      </div>
    </div>
  );
}

