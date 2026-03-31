import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { PostMeta } from '../types'

export default function Post() {
  const { slug } = useParams<{ slug: string }>()
  const navigate = useNavigate()
  const [content, setContent] = useState('')
  const [meta, setMeta] = useState<PostMeta | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load index to find the file
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
            setContent(stripped)
            setLoading(false)
          })
      })
      .catch(() => setLoading(false))
  }, [slug])

  if (loading) return <div className="post-page"><p>Loading...</p></div>
  if (!content) return (
    <div className="post-page empty">
      <h2>Post not found</h2>
      <p><a href="#" onClick={() => navigate('/')}>← Back to home</a></p>
    </div>
  )

  return (
    <div className="post-page">
      <div style={{ marginBottom: 24 }}>
        <a href="#" onClick={e => { e.preventDefault(); navigate('/') }}
          style={{ fontSize: '0.82rem', fontFamily: 'sans-serif', color: '#6b7280' }}>
          ← All reports
        </a>
      </div>

      {meta && (
        <div className="post-page-meta">
          <span>📅 {meta.date}</span>
          <span>🗂 {meta.dataset}</span>
          <span>👥 {meta.subjects} subjects</span>
          {meta.mean_accuracy && (
            <span>🎯 {(parseFloat(meta.mean_accuracy) * 100).toFixed(0)}% accuracy</span>
          )}
        </div>
      )}

      <div className="post-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            img: ({ src, alt }) => <img src={src && src.startsWith('/figures/') ? `/eeg-insights${src}` : src} alt={alt ?? ''} style={{ maxWidth: '100%', borderRadius: 6 }} />
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  )
}
