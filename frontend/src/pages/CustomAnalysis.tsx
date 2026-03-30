import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { PostMeta } from '../types'

const GH_TOKEN  = import.meta.env.VITE_GH_TOKEN as string | undefined
const REPO      = 'yuriao/eeg-insights'
const BRANCH    = 'main'

const DEFAULT_CODE = `# Custom EEG Analysis Function
# ─────────────────────────────────────────────────────────────────────────
# This function is called by the pipeline after the standard ERP/ERDS/CSP
# analysis. Use it to add your own figures or metrics.
#
# Signature (must keep exactly):
#   def custom_analysis(epochs, slug: str, figures_dir: Path) -> dict
#
# Return a dict with at least:
#   {
#     "figure_path": "/figures/<slug>_custom.png",  # relative web path
#     "summary": "One-line summary shown in the report",
#     "metrics": { "key": value, ... }               # optional extra metrics
#   }

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def custom_analysis(epochs, slug: str, figures_dir: Path) -> dict:
    """Example: plot per-channel variance across conditions."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')

    for cond in list(epochs.event_id.keys())[:4]:
        data = epochs[cond].get_data()          # (trials, channels, times)
        var  = data.var(axis=(0, 2))            # variance per channel
        ax.plot(var, label=cond, lw=1.5, alpha=0.8)

    ax.set_xlabel('Channel index')
    ax.set_ylabel('Variance (V²)')
    ax.set_title('Per-channel signal variance by condition')
    ax.legend(fontsize=8)

    out_path = figures_dir / f'{slug}_custom.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    return {
        "figure_path": f"/figures/{slug}_custom.png",
        "summary": "Per-channel variance analysis completed",
        "metrics": {"max_variance_channel": int(var.argmax())}
    }
`

const DATASETS = [
  { id: 'BNCI2014_001', name: 'BCI Competition IV 2a (9 subjects)' },
  { id: 'BNCI2014_004', name: 'BCI Competition IV 2b (9 subjects)' },
  { id: 'BNCI2015_001', name: 'BNCI 2015-001 (12 subjects)' },
  { id: 'Zhou2016',     name: 'Zhou 2016 Motor Imagery (4 subjects)' },
]

interface SavedAnalysis {
  name: string
  description: string
  code: string
  created_at: string
}

type SavedMap = Record<string, SavedAnalysis>

async function loadSaved(): Promise<SavedMap> {
  try {
    const res = await fetch(
      `https://api.github.com/repos/${REPO}/contents/pipeline/custom_analyses.json`,
      GH_TOKEN ? { headers: { Authorization: `Bearer ${GH_TOKEN}` } } : {}
    )
    if (!res.ok) return {}
    const data = await res.json()
    return JSON.parse(atob(data.content.replace(/\n/g, '')))
  } catch {
    return {}
  }
}

async function saveAnalysis(
  name: string, description: string, code: string, existing: SavedMap
): Promise<{ ok: boolean; message: string }> {
  if (!GH_TOKEN) return { ok: false, message: 'GitHub token not configured.' }

  const updated: SavedMap = {
    ...existing,
    [name]: { name, description, code, created_at: new Date().toISOString() },
  }
  const content = btoa(unescape(encodeURIComponent(JSON.stringify(updated, null, 2))))

  // Get current SHA if file exists
  let sha: string | undefined
  try {
    const r = await fetch(
      `https://api.github.com/repos/${REPO}/contents/pipeline/custom_analyses.json`,
      { headers: { Authorization: `Bearer ${GH_TOKEN}` } }
    )
    if (r.ok) { const d = await r.json(); sha = d.sha }
  } catch {}

  const body: Record<string, string> = {
    message: `feat: save custom analysis "${name}"`,
    content,
    branch: BRANCH,
  }
  if (sha) body.sha = sha

  const res = await fetch(
    `https://api.github.com/repos/${REPO}/contents/pipeline/custom_analyses.json`,
    {
      method: 'PUT',
      headers: {
        Authorization: `Bearer ${GH_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }
  )
  return res.ok
    ? { ok: true, message: `Saved "${name}" successfully.` }
    : { ok: false, message: `Save failed: ${res.status}` }
}

async function triggerRun(
  dataset: string, analysisName: string, rerunSlug?: string
): Promise<{ ok: boolean; message: string }> {
  if (!GH_TOKEN) return { ok: false, message: 'GitHub token not configured.' }

  const inputs: Record<string, string> = {
    dataset,
    custom_analysis: analysisName,
  }
  if (rerunSlug) inputs.rerun_slug = rerunSlug

  const res = await fetch(
    `https://api.github.com/repos/${REPO}/actions/workflows/analyze.yml/dispatches`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${GH_TOKEN}`,
        Accept: 'application/vnd.github+json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ref: BRANCH, inputs }),
    }
  )
  return res.status === 204
    ? { ok: true, message: 'Analysis triggered! Results will appear in ~3 minutes.' }
    : { ok: false, message: `Trigger failed: ${res.status}` }
}

export default function CustomAnalysis() {
  const [saved, setSaved]           = useState<SavedMap>({})
  const [code, setCode]             = useState(DEFAULT_CODE)
  const [name, setName]             = useState('')
  const [description, setDesc]      = useState('')
  const [selectedSaved, setSelected]= useState<string>('')
  const [dataset, setDataset]       = useState('BNCI2015_001')
  const [posts, setPosts]           = useState<PostMeta[]>([])
  const [rerunSlug, setRerunSlug]   = useState('')
  const [status, setStatus]         = useState<{ ok: boolean; text: string } | null>(null)
  const [saving, setSaving]         = useState(false)
  const [running, setRunning]       = useState(false)
  const [tab, setTab]               = useState<'editor' | 'saved'>('editor')

  useEffect(() => {
    loadSaved().then(setSaved)
    fetch('/eeg-insights/posts-index.json')
      .then(r => r.json())
      .then(setPosts)
      .catch(() => {})
  }, [])

  const handleSave = async () => {
    if (!name.trim()) { setStatus({ ok: false, text: 'Please enter a name.' }); return }
    setSaving(true); setStatus(null)
    const res = await saveAnalysis(name.trim(), description.trim(), code, saved)
    setStatus({ ok: res.ok, text: res.message })
    if (res.ok) {
      const updated = await loadSaved()
      setSaved(updated)
    }
    setSaving(false)
  }

  const handleRun = async () => {
    if (!name.trim()) { setStatus({ ok: false, text: 'Save the analysis first.' }); return }
    setRunning(true); setStatus(null)
    const res = await triggerRun(dataset, name.trim(), rerunSlug || undefined)
    setStatus({ ok: res.ok, text: res.message })
    setRunning(false)
  }

  const loadSavedAnalysis = (key: string) => {
    const a = saved[key]
    if (!a) return
    setCode(a.code)
    setName(a.name)
    setDesc(a.description)
    setSelected(key)
    setTab('editor')
  }

  return (
    <div className="home">
      <div className="home-hero">
        <div style={{ marginBottom: 8 }}>
          <Link to="/" style={{ fontSize: '0.85rem', fontFamily: 'sans-serif', color: '#6b7280' }}>
            ← Back to reports
          </Link>
          {' · '}
          <Link to="/algorithms" style={{ fontSize: '0.85rem', fontFamily: 'sans-serif', color: '#6b7280' }}>
            Algorithm reference
          </Link>
        </div>
        <h1>Custom Analysis Editor</h1>
        <p>
          Write a Python function, save it to the repo, then trigger a full pipeline 
          run — against any dataset or existing report.
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '2px solid #f0f0f0', marginBottom: 24 }}>
        {(['editor', 'saved'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '10px 24px', border: 'none', background: 'none',
              fontFamily: 'sans-serif', fontSize: '0.9rem', fontWeight: 600,
              color: tab === t ? '#1a6b3c' : '#6b7280',
              borderBottom: tab === t ? '2px solid #1a6b3c' : '2px solid transparent',
              cursor: 'pointer', marginBottom: -2,
            }}
          >
            {t === 'editor' ? '✏️ Editor' : `📁 Saved (${Object.keys(saved).length})`}
          </button>
        ))}
      </div>

      {tab === 'saved' && (
        <div>
          {Object.keys(saved).length === 0 ? (
            <div className="empty" style={{ textAlign: 'center', padding: '40px 0' }}>
              <p style={{ fontFamily: 'sans-serif', color: '#6b7280' }}>
                No saved analyses yet. Write one in the Editor tab.
              </p>
            </div>
          ) : (
            Object.values(saved).map(a => (
              <div
                key={a.name}
                style={{
                  padding: '16px 0', borderBottom: '1px solid #f0f0f0',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
                }}
              >
                <div>
                  <div style={{ fontWeight: 700, marginBottom: 4 }}>{a.name}</div>
                  {a.description && (
                    <div style={{ fontFamily: 'sans-serif', fontSize: '0.85rem', color: '#6b7280', marginBottom: 4 }}>
                      {a.description}
                    </div>
                  )}
                  <div style={{ fontFamily: 'sans-serif', fontSize: '0.75rem', color: '#9ca3af' }}>
                    Saved {new Date(a.created_at).toLocaleDateString()}
                  </div>
                </div>
                <button
                  onClick={() => loadSavedAnalysis(a.name)}
                  style={{
                    padding: '6px 16px', background: '#ecfdf5', color: '#1a6b3c',
                    border: '1px solid #a7f3d0', borderRadius: 6,
                    fontFamily: 'sans-serif', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  Load
                </button>
              </div>
            ))
          )}
        </div>
      )}

      {tab === 'editor' && (
        <div>
          {/* Name + description */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 200 }}>
              <label style={{ fontFamily: 'sans-serif', fontSize: '0.8rem', fontWeight: 600, color: '#374151', display: 'block', marginBottom: 4 }}>
                Analysis name *
              </label>
              <input
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="e.g. channel_variance"
                style={{
                  width: '100%', padding: '8px 12px',
                  border: '1px solid #d1d5db', borderRadius: 6,
                  fontFamily: 'sans-serif', fontSize: '0.9rem',
                }}
              />
            </div>
            <div style={{ flex: 2, minWidth: 280 }}>
              <label style={{ fontFamily: 'sans-serif', fontSize: '0.8rem', fontWeight: 600, color: '#374151', display: 'block', marginBottom: 4 }}>
                Description
              </label>
              <input
                value={description}
                onChange={e => setDesc(e.target.value)}
                placeholder="What does this analysis do?"
                style={{
                  width: '100%', padding: '8px 12px',
                  border: '1px solid #d1d5db', borderRadius: 6,
                  fontFamily: 'sans-serif', fontSize: '0.9rem',
                }}
              />
            </div>
          </div>

          {/* Code editor */}
          <div style={{ marginBottom: 20 }}>
            <div style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              background: '#1e1e2e', borderRadius: '8px 8px 0 0', padding: '8px 16px',
            }}>
              <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: '#6b7280' }}>
                Python · custom_analysis(epochs, slug, figures_dir) → dict
              </span>
              <button
                onClick={() => setCode(DEFAULT_CODE)}
                style={{
                  background: 'transparent', border: '1px solid #374151',
                  color: '#9ca3af', borderRadius: 4, padding: '2px 10px',
                  fontSize: '0.75rem', fontFamily: 'sans-serif', cursor: 'pointer',
                }}
              >
                Reset
              </button>
            </div>
            <textarea
              value={code}
              onChange={e => setCode(e.target.value)}
              spellCheck={false}
              style={{
                width: '100%', minHeight: 420,
                background: '#1e1e2e', color: '#cdd6f4',
                border: 'none', borderRadius: '0 0 8px 8px',
                padding: 16, fontSize: '0.8rem', lineHeight: 1.6,
                fontFamily: '"Fira Code", "JetBrains Mono", monospace',
                resize: 'vertical', outline: 'none',
              }}
            />
          </div>

          {/* Save button */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 32, flexWrap: 'wrap' }}>
            <button
              onClick={handleSave}
              disabled={saving}
              style={{
                padding: '10px 24px', background: saving ? '#f3f4f6' : '#1a6b3c',
                color: saving ? '#9ca3af' : 'white',
                border: 'none', borderRadius: 6,
                fontFamily: 'sans-serif', fontSize: '0.9rem', fontWeight: 600,
                cursor: saving ? 'not-allowed' : 'pointer',
              }}
            >
              {saving ? 'Saving…' : '💾 Save to Repo'}
            </button>
          </div>

          {/* Run section */}
          <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 24 }}>
            <div style={{ fontWeight: 700, marginBottom: 16, fontSize: '1rem' }}>Run Analysis</div>
            <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: 220 }}>
                <label style={{ fontFamily: 'sans-serif', fontSize: '0.8rem', fontWeight: 600, color: '#374151', display: 'block', marginBottom: 4 }}>
                  Dataset
                </label>
                <select
                  value={dataset}
                  onChange={e => setDataset(e.target.value)}
                  style={{
                    width: '100%', padding: '8px 12px',
                    border: '1px solid #d1d5db', borderRadius: 6,
                    fontFamily: 'sans-serif', fontSize: '0.85rem', background: 'white',
                  }}
                >
                  {DATASETS.map(d => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                </select>
              </div>

              <div style={{ flex: 1, minWidth: 220 }}>
                <label style={{ fontFamily: 'sans-serif', fontSize: '0.8rem', fontWeight: 600, color: '#374151', display: 'block', marginBottom: 4 }}>
                  Re-run existing report (optional)
                </label>
                <select
                  value={rerunSlug}
                  onChange={e => setRerunSlug(e.target.value)}
                  style={{
                    width: '100%', padding: '8px 12px',
                    border: '1px solid #d1d5db', borderRadius: 6,
                    fontFamily: 'sans-serif', fontSize: '0.85rem', background: 'white',
                  }}
                >
                  <option value="">— New run —</option>
                  {posts.map(p => (
                    <option key={p.slug} value={p.slug}>
                      {p.date} · {p.dataset}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button
              onClick={handleRun}
              disabled={running || !name.trim()}
              style={{
                padding: '10px 28px',
                background: running || !name.trim() ? '#f3f4f6' : '#0f4c35',
                color: running || !name.trim() ? '#9ca3af' : '#4ade80',
                border: '1px solid currentColor', borderRadius: 6,
                fontFamily: 'sans-serif', fontSize: '0.9rem', fontWeight: 600,
                cursor: running || !name.trim() ? 'not-allowed' : 'pointer',
              }}
            >
              {running ? '⏳ Triggering…' : '▶ Run Pipeline'}
            </button>
            <div style={{ fontFamily: 'sans-serif', fontSize: '0.75rem', color: '#9ca3af', marginTop: 6 }}>
              Save the analysis first, then run it.
            </div>
          </div>

          {status && (
            <div style={{
              marginTop: 16, padding: '10px 16px', borderRadius: 6,
              background: status.ok ? '#f0fdf4' : '#fef2f2',
              color: status.ok ? '#166534' : '#991b1b',
              fontFamily: 'sans-serif', fontSize: '0.85rem', fontWeight: 500,
            }}>
              {status.ok ? '✅' : '❌'} {status.text}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
