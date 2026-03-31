import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { PostMeta } from '../types'

const GH_TOKEN = import.meta.env.VITE_GH_TOKEN as string | undefined

async function triggerAnalysis(): Promise<{ ok: boolean; message: string }> {
  if (!GH_TOKEN) {
    return { ok: false, message: 'Trigger token not configured.' }
  }
  const res = await fetch(
    'https://api.github.com/repos/yuriao/eeg-insights/actions/workflows/analyze.yml/dispatches',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${GH_TOKEN}`,
        Accept: 'application/vnd.github+json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ref: 'main' }),
    }
  )
  if (res.status === 204) {
    return { ok: true, message: 'Analysis triggered! Results will appear in ~3 minutes.' }
  }
  const err = await res.json().catch(() => ({}))
  return { ok: false, message: (err as { message?: string }).message ?? `Error ${res.status}` }
}

export default function Home() {
  const [posts, setPosts] = useState<PostMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [triggering, setTriggering] = useState(false)
  const [triggerMsg, setTriggerMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    fetch('/eeg-insights/posts-index.json')
      .then(r => r.json())
      .then(data => { setPosts(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const acc = (p: PostMeta) => parseFloat(p.mean_accuracy)
  const isGood = (p: PostMeta) => acc(p) >= 0.7

  const handleTrigger = async () => {
    setTriggering(true)
    setTriggerMsg(null)
    const result = await triggerAnalysis()
    setTriggerMsg({ ok: result.ok, text: result.message })
    setTriggering(false)
    if (result.ok) setTimeout(() => setTriggerMsg(null), 8000)
  }

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
              🎯 Avg {posts.filter(p => p.mean_accuracy).length > 0
                ? (posts.filter(p => p.mean_accuracy).reduce((s, p) => s + acc(p), 0) / posts.filter(p => p.mean_accuracy).length * 100).toFixed(0)
                : '—'}% accuracy
            </div>
          </div>
        )}

        <div style={{ marginTop: 24 }}>
          <button
            onClick={handleTrigger}
            disabled={triggering}
            style={{
              padding: '10px 24px',
              background: triggering ? '#1e293b' : '#0f4c35',
              color: triggering ? '#64748b' : '#4ade80',
              border: '1px solid currentColor',
              borderRadius: '8px',
              fontSize: '0.9rem',
              fontWeight: 600,
              cursor: triggering ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s',
            }}
          >
            {triggering ? '⏳ Triggering…' : '▶ Run Analysis Now'}
          </button>
          {triggerMsg && (
            <div style={{
              marginTop: 10,
              padding: '8px 14px',
              borderRadius: '6px',
              fontSize: '0.85rem',
              background: triggerMsg.ok ? '#052e16' : '#2d0a0a',
              color: triggerMsg.ok ? '#86efac' : '#fca5a5',
              display: 'inline-block',
            }}>
              {triggerMsg.ok ? '✅' : '❌'} {triggerMsg.text}
            </div>
          )}
        </div>
      </div>

      <div className="section-title">Latest Reports</div>

      {loading && <div className="empty"><p>Loading reports...</p></div>}

      {!loading && posts.length === 0 && (
        <div className="empty">
          <h2>No reports yet</h2>
          <p>
            The analysis pipeline runs weekly via GitHub Actions.<br />
            Use the button above to trigger it manually.
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
              {post.paradigm === 'preprocessing'
                ? 'Signal preprocessing · Filtering · ICA · Artifact removal · Epoching'
                : 'ERP analysis · ERDS time-frequency maps · CSP+LDA decoding'}
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
              {post.paradigm === 'preprocessing' && (
                <span className="post-badge" style={{ background: '#eff6ff', color: '#1d4ed8' }}>
                  📖 Tutorial
                </span>
              )}
              {post.mean_accuracy && post.paradigm !== 'preprocessing' && (
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

