import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import type { PostMeta } from '../types'

// ── Cell types ────────────────────────────────────────────────────────────────

interface MarkdownCell {
  type: 'markdown'
  content: string
}

interface CodeCell {
  type: 'code'
  language: string
  content: string
  index: number   // execution order label (In [N]:)
}

type Cell = MarkdownCell | CodeCell

// ── Parse markdown into Jupyter-style cells ───────────────────────────────────
// Fenced code blocks (```python ... ```) become CodeCells.
// Everything between them becomes MarkdownCells.

function parseCells(md: string): Cell[] {
  const cells: Cell[] = []
  // Regex: matches ``` optionally followed by language name, then content, then ```
  const fenceRe = /^```(\w*)\n([\s\S]*?)^```/gm
  let lastIndex = 0
  let codeCount = 0

  for (const match of md.matchAll(fenceRe)) {
    const lang    = match[1] || 'text'
    const code    = match[2]
    const start   = match.index!
    const end     = start + match[0].length

    // Text before this code block → markdown cell
    const before = md.slice(lastIndex, start).trim()
    if (before) cells.push({ type: 'markdown', content: before })

    // Code block → code cell
    codeCount++
    cells.push({ type: 'code', language: lang, content: code.replace(/\n$/, ''), index: codeCount })
    lastIndex = end + 1  // skip trailing newline
  }

  // Remaining text after last code block
  const after = md.slice(lastIndex).trim()
  if (after) cells.push({ type: 'markdown', content: after })

  return cells
}

// ── Cell components ───────────────────────────────────────────────────────────

function MarkdownCellView({ content }: { content: string }) {
  return (
    <div className="nb-markdown-cell">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          img: ({ src, alt }) => (
            <img
              src={src && src.startsWith('/figures/') ? `/eeg-insights${src}` : src}
              alt={alt ?? ''}
              style={{ maxWidth: '100%', borderRadius: 6 }}
            />
          ),
          // Inline code (backtick) stays as inline, not a full cell
          code: ({ children, className }) => {
            const isInline = !className
            if (isInline) return <code className="nb-inline-code">{children}</code>
            // Shouldn't reach here (fenced blocks are handled by parseCells)
            return <code>{children}</code>
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

function CodeCellView({ cell }: { cell: CodeCell }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(cell.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="nb-code-cell">
      {/* Left gutter — In [N]: prompt */}
      <div className="nb-prompt">
        <span className="nb-prompt-label">In&nbsp;[{cell.index}]:</span>
      </div>

      {/* Code area */}
      <div className="nb-code-area">
        <div className="nb-code-toolbar">
          <span className="nb-lang-badge">{cell.language || 'code'}</span>
          <button className="nb-copy-btn" onClick={handleCopy} title="Copy code">
            {copied ? '✓ Copied' : '⎘ Copy'}
          </button>
        </div>
        <SyntaxHighlighter
          language={cell.language === 'py' ? 'python' : (cell.language || 'text')}
          style={oneLight}
          customStyle={{
            margin: 0,
            borderRadius: '0 0 6px 6px',
            fontSize: '0.84rem',
            background: '#f8f8f8',
            border: 'none',
          }}
          showLineNumbers={cell.content.split('\n').length > 5}
          lineNumberStyle={{ color: '#bbb', fontSize: '0.75rem', minWidth: '2.5em' }}
        >
          {cell.content}
        </SyntaxHighlighter>
      </div>
    </div>
  )
}

// ── Notebook toolbar (Colab + download buttons) ───────────────────────────────

function NotebookToolbar({ meta: _meta, slug }: { meta: PostMeta; slug: string }) {
  const notebookUrl = `https://raw.githubusercontent.com/yuriao/eeg-insights/main/notebooks/${slug}.ipynb`
  const colabUrl    = `https://colab.research.google.com/github/yuriao/eeg-insights/blob/main/notebooks/${slug}.ipynb`

  return (
    <div className="nb-toolbar">
      <div className="nb-toolbar-left">
        <span className="nb-kernel-badge">🐍 Python 3</span>
        <span className="nb-kernel-badge">MNE · MOABB · sklearn</span>
      </div>
      <div className="nb-toolbar-right">
        <a
          href={colabUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="nb-colab-btn"
          title="Open in Google Colab"
        >
          <img
            src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open In Colab"
            style={{ height: 20 }}
          />
        </a>
        <a
          href={notebookUrl}
          download={`${slug}.ipynb`}
          className="nb-download-btn"
          title="Download .ipynb"
        >
          ⬇ .ipynb
        </a>
      </div>
    </div>
  )
}

// ── Main Post component ───────────────────────────────────────────────────────

export default function Post() {
  const { slug }   = useParams<{ slug: string }>()
  const navigate   = useNavigate()
  const [cells, setCells]   = useState<Cell[]>([])
  const [meta, setMeta]     = useState<PostMeta | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/eeg-insights/posts-index.json')
      .then(r => r.json())
      .then((posts: PostMeta[]) => {
        const found = posts.find(p => p.slug === slug)
        if (!found) { setLoading(false); return }
        setMeta(found)
        return fetch(`/eeg-insights/posts/${found.file}`)
          .then(r => r.text())
          .then(text => {
            // Strip YAML frontmatter
            const stripped = text.replace(/^---[\s\S]*?---\n/, '')
            setCells(parseCells(stripped))
            setLoading(false)
          })
      })
      .catch(() => setLoading(false))
  }, [slug])

  if (loading) return (
    <div className="post-page">
      <div className="nb-loading">
        <span className="nb-loading-dot" />
        <span className="nb-loading-dot" />
        <span className="nb-loading-dot" />
      </div>
    </div>
  )

  if (!cells.length) return (
    <div className="post-page empty">
      <h2>Post not found</h2>
      <p><a href="#" onClick={() => navigate('/')}>← Back to home</a></p>
    </div>
  )

  return (
    <div className="post-page nb-page">
      {/* Back link */}
      <a
        href="#"
        onClick={e => { e.preventDefault(); navigate('/') }}
        className="nb-back-link"
      >
        ← All reports
      </a>

      {/* Jupyter-style notebook toolbar */}
      {meta && <NotebookToolbar meta={meta} slug={slug!} />}

      {/* Post metadata bar */}
      {meta && (
        <div className="post-page-meta">
          <span>📅 {meta.date}</span>
          <span>🗂 {meta.dataset}</span>
          {meta.subjects && <span>👥 {meta.subjects} subjects</span>}
          {meta.mean_accuracy && (
            <span>🎯 {(parseFloat(meta.mean_accuracy) * 100).toFixed(0)}% accuracy</span>
          )}
          {(meta as any).algorithm && (
            <span>⚙️ {(meta as any).algorithm}</span>
          )}
        </div>
      )}

      {/* Notebook cells */}
      <div className="nb-cells">
        {cells.map((cell, i) =>
          cell.type === 'markdown'
            ? <MarkdownCellView key={i} content={cell.content} />
            : <CodeCellView    key={i} cell={cell} />
        )}
      </div>
    </div>
  )
}
