import { useState } from 'react'
import { Link } from 'react-router-dom'

const ALGORITHMS = [
  {
    id: 'erp',
    name: 'Event-Related Potential (ERP)',
    icon: '📈',
    tagline: 'Time-domain averaging to extract neural responses locked to stimuli',
    description: `ERP analysis averages EEG signals across many trials locked to a stimulus or 
event onset. Because background brain noise is random, it cancels out with averaging, 
leaving the consistent neural response. The result reveals characteristic waveform 
components (P300, N200, mu-suppression) that reflect specific cognitive or motor processes.

**When to use:** Any paradigm with discrete stimulus events — P300 spellers, motor 
imagery onset, auditory/visual evoked potentials.

**Key parameters:**
- **tmin / tmax** — epoch window around the event
- **Baseline** — pre-stimulus period used to normalize the signal
- **Channel selection** — occipital for visual, central for motor, frontal for cognitive`,
    code: `def plot_erp(epochs, slug: str) -> str:
    """Grand-average ERP for each condition."""
    fig, axes = plt.subplots(
        1, len(epochs.event_id),
        figsize=(4 * len(epochs.event_id), 3.5),
        facecolor='white', constrained_layout=True
    )
    if len(epochs.event_id) == 1:
        axes = [axes]

    for ax, (cond, _) in zip(axes, epochs.event_id.items()):
        evoked = epochs[cond].average()
        times  = evoked.times
        data   = evoked.data * 1e6        # V → µV
        picks  = list(range(min(5, len(evoked.ch_names))))

        for ch_idx in picks:
            ax.plot(times, data[ch_idx], lw=1.2, alpha=0.8,
                    label=evoked.ch_names[ch_idx])

        ax.axvline(0, color='k', lw=0.8, ls='--')   # stimulus onset
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(f'ERP — {cond}', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('µV')
        ax.legend(fontsize=7, loc='upper right')

    path = FIGURES_DIR / f'{slug}_erp.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return f'/figures/{slug}_erp.png'`,
  },
  {
    id: 'erds',
    name: 'Event-Related Desynchronization / Synchronization (ERDS)',
    icon: '🌊',
    tagline: 'Time-frequency decomposition revealing spectral power dynamics',
    description: `ERDS measures changes in oscillatory power relative to a baseline period.
Desynchronization (ERD) — a power decrease — indicates activated brain regions, 
while synchronization (ERS) — a power increase — reflects inhibited or rebounding areas.

In motor imagery, contralateral alpha (8–12 Hz) and beta (13–30 Hz) bands show clear ERD 
during imagination, followed by post-movement beta rebound (ERS).

**When to use:** Motor imagery, movement preparation, attention, working memory — 
any task involving rhythmic neural oscillations.

**Key parameters:**
- **Frequency range** — alpha (8–12 Hz), beta (13–30 Hz), gamma (30–80 Hz)
- **n_cycles** — controls time-frequency trade-off (more cycles = better freq resolution)
- **Baseline window** — pre-stimulus period for normalization`,
    code: `def plot_erds(epochs, slug: str) -> str:
    """ERDS time-frequency map using multitaper method."""
    from mne.time_frequency import tfr_multitaper

    freqs    = np.arange(4, 40, 2)     # 4–40 Hz in 2 Hz steps
    n_cycles = freqs / 2.              # adaptive: more cycles at higher freqs

    fig, axes = plt.subplots(
        1, min(len(epochs.event_id), 4),
        figsize=(14, 3.5), facecolor='white', constrained_layout=True
    )
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax, (cond, _) in zip(axes, list(epochs.event_id.items())[:4]):
        power = tfr_multitaper(
            epochs[cond], freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False
        )
        avg_power = power.data[:3].mean(axis=0)   # average over first 3 channels
        im = ax.imshow(
            avg_power, aspect='auto', origin='lower',
            extent=[epochs.tmin, epochs.tmax, freqs[0], freqs[-1]],
            cmap='RdBu_r',
            vmin=np.percentile(avg_power, 5),
            vmax=np.percentile(avg_power, 95)
        )
        ax.axvline(0, color='k', lw=1, ls='--')
        ax.set_title(f'ERDS — {cond}', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    path = FIGURES_DIR / f'{slug}_erds.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return f'/figures/{slug}_erds.png'`,
  },
  {
    id: 'csp',
    name: 'Common Spatial Patterns + LDA (CSP+LDA)',
    icon: '🎯',
    tagline: 'Spatial filtering maximizing variance ratio between classes for decoding',
    description: `CSP finds spatial filters (linear combinations of EEG channels) that 
maximize variance for one class while minimizing it for another. The resulting components 
capture the spatially distinct neural signatures of each condition. Log-variance features 
from these components are then classified using Linear Discriminant Analysis (LDA).

CSP+LDA is the gold standard for motor imagery BCI decoding — simple, fast, 
interpretable, and remarkably effective when signal-to-noise ratio is good.

**When to use:** Any binary (or multi-class via OVR) classification of EEG epochs — 
motor imagery, mental workload, P300 detection.

**Key parameters:**
- **n_components** — number of CSP filters (typically 4–8)
- **reg** — regularization for ill-conditioned covariance (None, 'ledoit_wolf')
- **log** — apply log to variance features (almost always True)
- **CV folds** — StratifiedKFold 5 is standard for balanced evaluation`,
    code: `def decode_csp_lda(epochs) -> dict:
    """CSP + LDA decoding with 5-fold stratified cross-validation."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    X = epochs.get_data()          # shape: (n_trials, n_channels, n_times)
    y = epochs.events[:, 2]        # class labels

    # Keep two most common classes (binary decoding)
    classes, counts = np.unique(y, return_counts=True)
    top2 = classes[np.argsort(counts)[-2:]]
    mask = np.isin(y, top2)
    X, y = X[mask], y[mask]

    pipeline = Pipeline([
        ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ('lda', LinearDiscriminantAnalysis()),
    ])
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    return {
        'mean_accuracy': float(scores.mean()),
        'std_accuracy':  float(scores.std()),
        'n_trials':      len(y),
        'classes':       [str(c) for c in top2],
    }`,
  },
]

export default function Algorithms() {
  const [active, setActive] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)

  const copy = (id: string, code: string) => {
    navigator.clipboard.writeText(code)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }

  return (
    <div className="home">
      <div className="home-hero">
        <div style={{ marginBottom: 8 }}>
          <Link to="/" style={{ fontSize: '0.85rem', fontFamily: 'sans-serif', color: '#6b7280' }}>
            ← Back to reports
          </Link>
        </div>
        <h1>Analysis Algorithms</h1>
        <p>
          The pipeline uses three complementary EEG analysis methods. Each captures a
          different aspect of brain activity — time domain, frequency domain, and
          decoding performance.
        </p>
        <div style={{ marginTop: 16, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {ALGORITHMS.map(a => (
            <button
              key={a.id}
              onClick={() => setActive(active === a.id ? null : a.id)}
              style={{
                padding: '6px 16px',
                borderRadius: 20,
                border: '1px solid',
                borderColor: active === a.id ? '#1a6b3c' : '#d1d5db',
                background: active === a.id ? '#ecfdf5' : 'white',
                color: active === a.id ? '#1a6b3c' : '#374151',
                fontFamily: 'sans-serif',
                fontSize: '0.85rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              {a.icon} {a.id.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {ALGORITHMS.map((algo) => {
          const isOpen = active === null || active === algo.id
          return (
            <div
              key={algo.id}
              style={{
                borderBottom: '1px solid #f0f0f0',
                padding: '28px 0',
                opacity: active !== null && active !== algo.id ? 0.4 : 1,
                transition: 'opacity 0.2s',
              }}
            >
              <div
                style={{ cursor: 'pointer' }}
                onClick={() => setActive(active === algo.id ? null : algo.id)}
              >
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 6 }}>
                  <span style={{ fontSize: '1.4rem' }}>{algo.icon}</span>
                  <span style={{ fontSize: '1.1rem', fontWeight: 700 }}>{algo.name}</span>
                </div>
                <div style={{ fontFamily: 'sans-serif', fontSize: '0.85rem', color: '#6b7280', marginLeft: 36 }}>
                  {algo.tagline}
                </div>
              </div>

              {isOpen && (
                <div style={{ marginTop: 20, marginLeft: 36 }}>
                  {/* Description */}
                  <div style={{ fontFamily: 'sans-serif', fontSize: '0.9rem', color: '#374151', lineHeight: 1.7, marginBottom: 20, maxWidth: 680 }}>
                    {algo.description.split('\n').map((line, i) => {
                      if (line.startsWith('**') && line.endsWith('**')) {
                        return <p key={i} style={{ fontWeight: 700, marginTop: 14, marginBottom: 4 }}>{line.replace(/\*\*/g, '')}</p>
                      }
                      if (line.startsWith('- **')) {
                        const [label, ...rest] = line.replace('- **', '').split('**')
                        return (
                          <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
                            <span style={{ color: '#9ca3af' }}>·</span>
                            <span><strong>{label}</strong>{rest.join('')}</span>
                          </div>
                        )
                      }
                      if (line.trim() === '') return <br key={i} />
                      return <p key={i}>{line}</p>
                    })}
                  </div>

                  {/* Code block */}
                  <div style={{ position: 'relative' }}>
                    <div style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      background: '#1e1e2e', borderRadius: '8px 8px 0 0',
                      padding: '8px 16px',
                    }}>
                      <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: '#6b7280' }}>
                        pipeline/analyze.py
                      </span>
                      <button
                        onClick={() => copy(algo.id, algo.code)}
                        style={{
                          background: 'transparent', border: '1px solid #374151',
                          color: copied === algo.id ? '#4ade80' : '#9ca3af',
                          borderRadius: 4, padding: '2px 10px',
                          fontSize: '0.75rem', fontFamily: 'sans-serif', cursor: 'pointer',
                        }}
                      >
                        {copied === algo.id ? '✓ Copied' : 'Copy'}
                      </button>
                    </div>
                    <pre style={{
                      background: '#1e1e2e', color: '#cdd6f4',
                      padding: '16px', borderRadius: '0 0 8px 8px',
                      fontSize: '0.78rem', lineHeight: 1.6,
                      overflowX: 'auto', margin: 0,
                      fontFamily: '"Fira Code", "JetBrains Mono", monospace',
                    }}>
                      <code>{algo.code}</code>
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Custom analysis link */}
      <div style={{
        marginTop: 48, padding: '24px', background: '#f8fafc',
        borderRadius: 8, border: '1px solid #e2e8f0', textAlign: 'center',
      }}>
        <div style={{ fontSize: '1.5rem', marginBottom: 8 }}>🧪</div>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Want to run your own analysis?</div>
        <div style={{ fontFamily: 'sans-serif', fontSize: '0.85rem', color: '#6b7280', marginBottom: 16 }}>
          Write custom Python analysis code and run it against any dataset.
        </div>
        <Link
          to="/custom"
          style={{
            display: 'inline-block', padding: '8px 24px',
            background: '#1a6b3c', color: 'white',
            borderRadius: 6, fontFamily: 'sans-serif',
            fontSize: '0.85rem', fontWeight: 600,
          }}
        >
          Open Custom Analysis Editor →
        </Link>
      </div>
    </div>
  )
}
