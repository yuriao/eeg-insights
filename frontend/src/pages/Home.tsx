import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { PostMeta } from '../types'

export default function Home() {
  const [posts, setPosts] = useState<PostMeta[]>([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    fetch('/eeg-insights/posts-index.json')
      .then(r => r.json())
      .then(data => { setPosts(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const acc = (p: PostMeta) => parseFloat(p.mean_accuracy)
  const isGood = (p: PostMeta) => acc(p) >= 0.7

  return (
    <div className="home">
      <div className="home-hero">
        <h1>EEG Insights</h1>
        <p>
          Automated analysis reports on open-source EEG datasets. Each post runs
          ERP visualization, time-frequency ERDS mapping, and CSP+LDA decoding —
          generated weekly by a Python pipeline.
        </p>
        {posts.length > 0 && (
          <div className="stat-row" style={{ marginTop: 20 }}>
            <div className="stat-pill">📊 {posts.length} reports</div>
            <div className="stat-pill">🧠 {new Set(posts.map(p => p.dataset)).size} datasets</div>
            <div className="stat-pill">
              🎯 Avg {posts.length > 0
                ? (posts.reduce((s, p) => s + acc(p), 0) / posts.length * 100).toFixed(0)
                : '—'}% accuracy
            </div>
          </div>
        )}
      </div>

      <div className="section-title">Latest Reports</div>

      {loading && <div className="empty"><p>Loading reports...</p></div>}

      {!loading && posts.length === 0 && (
        <div className="empty">
          <h2>No reports yet</h2>
          <p>
            The analysis pipeline runs weekly via GitHub Actions.<br />
            Check back soon, or trigger it manually from the repo.
          </p>
        </div>
      )}

      {posts.map(post => (
        <div key={post.slug} className="post-card" onClick={() => navigate(`/post/${post.slug}`)}>
          <div className="post-card-body">
            <div className="post-meta">
              {post.date} · {post.dataset} · {post.subjects} subjects
            </div>
            <div className="post-title">{post.title}</div>
            <div className="post-excerpt">
              ERP analysis · ERDS time-frequency maps · CSP+LDA decoding
            </div>
            <div>
              {post.mean_accuracy && (
                <span className={`post-badge ${isGood(post) ? '' : 'warn'}`}>
                  {isGood(post) ? '✓' : '~'} {(acc(post) * 100).toFixed(0)}% decoding accuracy
                </span>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
