"""
EEG Insights — Analysis Pipeline
Runs one dataset through a rotating set of EEG analysis algorithms and generates a markdown post.

Usage:
  python analyze.py --dataset BNCI2014_001 --subject 1
  python analyze.py --auto           # picks next dataset + next unused algorithm
  python analyze.py --algo erp       # force specific algorithm
  python analyze.py --custom <name>  # run a saved custom analysis
  python analyze.py --rerun-slug <slug>  # re-run a specific report
"""

import os
import sys
import json
import random
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT       = Path(__file__).parent.parent
POSTS_DIR  = ROOT / 'posts'
FIGURES_DIR = ROOT / 'frontend' / 'public' / 'figures'
STATE_FILE  = ROOT / 'pipeline' / 'rotation_state.json'
POSTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Datasets ────────────────────────────────────────────────────────────────
DATASETS = [
    {'id': 'BNCI2014_001', 'name': 'BCI Competition IV 2a',   'paradigm': 'motor_imagery', 'subjects': 9},
    {'id': 'BNCI2014_004', 'name': 'BCI Competition IV 2b',   'paradigm': 'motor_imagery', 'subjects': 9},
    {'id': 'BNCI2015_001', 'name': 'BNCI 2015-001',           'paradigm': 'motor_imagery', 'subjects': 12},
    {'id': 'Zhou2016',     'name': 'Zhou 2016 Motor Imagery',  'paradigm': 'motor_imagery', 'subjects': 4},
    {'id': 'Schirrmeister2017', 'name': 'Schirrmeister 2017', 'paradigm': 'motor_imagery', 'subjects': 14},
]

# ── Algorithm registry ───────────────────────────────────────────────────────
# Each entry: id, name, year, description, run(epochs, slug) -> dict
ALGORITHMS = [
    {
        'id':   'csp_lda',
        'name': 'CSP + LDA',
        'year': 'Classic',
        'desc': 'Common Spatial Patterns with Linear Discriminant Analysis — the gold standard baseline for motor imagery BCI decoding.',
    },
    {
        'id':   'erp',
        'name': 'Event-Related Potential (ERP)',
        'year': 'Classic',
        'desc': 'Grand-average ERP visualization revealing time-locked neural responses across motor imagery conditions.',
    },
    {
        'id':   'erds',
        'name': 'ERDS Time-Frequency Mapping',
        'year': 'Classic',
        'desc': 'Event-Related Desynchronization/Synchronization via multitaper time-frequency decomposition, revealing alpha/beta spectral dynamics.',
    },
    {
        'id':   'riemann_mdm',
        'name': 'Riemannian Geometry — MDM Classifier',
        'year': '2010–2013 (widely adopted post-2021)',
        'desc': 'Minimum Distance to Mean classifier operating directly on the Riemannian manifold of covariance matrices. State-of-the-art for session/subject transfer without explicit feature engineering.',
    },
    {
        'id':   'fbcsp',
        'name': 'Filter Bank CSP (FBCSP)',
        'year': '2008, dominant benchmark 2021+',
        'desc': 'Applies CSP independently across multiple frequency sub-bands (delta, theta, alpha, beta, gamma), then selects the most discriminative features. Winner of BCI Competition IV.',
    },
    {
        'id':   'tangent_space_svm',
        'name': 'Tangent Space + SVM (TS-SVM)',
        'year': '2013, SOTA benchmark 2022+',
        'desc': 'Projects covariance matrices onto the tangent space of the Riemannian manifold at the Fréchet mean, then applies SVM. Achieves top performance on MOABB leaderboards.',
    },
    {
        'id':   'shrinkage_lda',
        'name': 'Shrinkage LDA (sLDA / SWLDA)',
        'year': 'Ledoit-Wolf 2004, popularised in EEG 2021+',
        'desc': 'Regularised LDA using Ledoit-Wolf shrinkage estimator for the covariance matrix — dramatically improves stability when n_samples < n_features, common in high-density EEG.',
    },
    {
        'id':   'xdawn_riemannian',
        'name': 'xDAWN + Riemannian (P300/ERP decoding)',
        'year': '2015–2022',
        'desc': 'xDAWN spatial filtering (designed for P300 ERP enhancement) followed by Riemannian covariance classification. Best-in-class for P300 speller paradigms.',
    },
]


# ── Rotation state ───────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {'algo_index': 0, 'dataset_index': 0, 'runs': []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def next_algo(state: dict) -> dict:
    idx = state.get('algo_index', 0) % len(ALGORITHMS)
    state['algo_index'] = idx + 1
    return ALGORITHMS[idx]


def next_dataset(state: dict) -> dict:
    idx = state.get('dataset_index', 0) % len(DATASETS)
    state['dataset_index'] = idx + 1
    return DATASETS[idx]


# ── MOABB helpers ─────────────────────────────────────────────────────────────
def load_dataset(dataset_id: str, subject: int):
    import moabb
    moabb.set_log_level('ERROR')
    dataset_map = {
        'BNCI2014_001': lambda: __import__('moabb.datasets', fromlist=['BNCI2014_001']).BNCI2014_001(),
        'BNCI2014_004': lambda: __import__('moabb.datasets', fromlist=['BNCI2014_004']).BNCI2014_004(),
        'BNCI2015_001': lambda: __import__('moabb.datasets', fromlist=['BNCI2015_001']).BNCI2015_001(),
        'Zhou2016':     lambda: __import__('moabb.datasets', fromlist=['Zhou2016']).Zhou2016(),
        'Schirrmeister2017': lambda: __import__('moabb.datasets', fromlist=['Schirrmeister2017']).Schirrmeister2017(),
    }
    ds = dataset_map[dataset_id]()
    sessions = ds.get_data(subjects=[subject])
    return ds, sessions


def get_epochs(sessions, subject, tmin=-0.5, tmax=2.5):
    import mne
    raws = []
    for sess_key, sess_val in sessions[subject].items():
        for run_key, raw in sess_val.items():
            raws.append(raw)
    raw = mne.concatenate_raws(raws)
    raw.filter(1, 40, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_id = {k: v for i, (k, v) in enumerate(event_id.items()) if i < 4}
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=(None, 0), preload=True, verbose=False)
    return epochs


# ── Algorithm implementations ─────────────────────────────────────────────────

def run_csp_lda(epochs, slug: str) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    X = epochs.get_data()
    y = epochs.events[:, 2]
    classes, counts = np.unique(y, return_counts=True)
    top2 = classes[np.argsort(counts)[-2:]]
    mask = np.isin(y, top2)
    X, y = X[mask], y[mask]

    pipeline = Pipeline([
        ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ('lda', LinearDiscriminantAnalysis()),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    # Plot per-fold accuracy
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white')
    folds = [f'Fold {i+1}' for i in range(len(scores))]
    ax.bar(folds, scores, color=['#3b82f6' if s >= 0.7 else '#94a3b8' for s in scores], edgecolor='white')
    ax.axhline(0.5, color='#ef4444', lw=1.2, ls='--', label='Chance')
    ax.axhline(scores.mean(), color='#1d4ed8', lw=1.2, ls=':', label=f'Mean={scores.mean():.2%}')
    ax.set_ylim(0, 1); ax.set_ylabel('Accuracy'); ax.set_title('CSP+LDA — 5-Fold Cross-Validation')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIGURES_DIR / f'{slug}_main.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight'); plt.close(fig)

    return {
        'mean_accuracy': float(scores.mean()),
        'std_accuracy': float(scores.std()),
        'figure_path': f'/figures/{slug}_main.png',
        'extra_md': f'| Fold | Accuracy |\n|------|----------|\n' +
                    '\n'.join(f'| {i+1} | {s:.1%} |' for i, s in enumerate(scores)),
    }


def run_erp(epochs, slug: str) -> dict:
    fig, axes = plt.subplots(1, len(epochs.event_id), figsize=(4 * len(epochs.event_id), 3.5),
                             facecolor='white', constrained_layout=True)
    if len(epochs.event_id) == 1:
        axes = [axes]
    for ax, (cond, _) in zip(axes, epochs.event_id.items()):
        evoked = epochs[cond].average()
        data = evoked.data * 1e6
        picks = list(range(min(5, len(evoked.ch_names))))
        for ch_idx in picks:
            ax.plot(evoked.times, data[ch_idx], lw=1.2, alpha=0.8, label=evoked.ch_names[ch_idx])
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(f'ERP — {cond}', fontsize=10)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('µV')
        ax.legend(fontsize=7)
    fig_path = FIGURES_DIR / f'{slug}_main.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight'); plt.close(fig)

    # Peak latency per condition
    peaks = {}
    for cond in epochs.event_id:
        evoked = epochs[cond].average()
        mean_signal = np.abs(evoked.data * 1e6).mean(axis=0)
        peak_idx = np.argmax(mean_signal)
        peaks[cond] = f'{evoked.times[peak_idx]*1000:.0f} ms ({mean_signal[peak_idx]:.1f} µV)'

    extra = '| Condition | Peak latency |\n|-----------|-------------|\n'
    extra += '\n'.join(f'| {cond} | {v} |' for cond, v in peaks.items())

    return {
        'mean_accuracy': 0.0,
        'std_accuracy': 0.0,
        'figure_path': f'/figures/{slug}_main.png',
        'extra_md': extra,
    }


def run_erds(epochs, slug: str) -> dict:
    from mne.time_frequency import tfr_multitaper
    freqs = np.arange(4, 40, 2)
    n_cycles = freqs / 2.
    fig, axes = plt.subplots(1, min(len(epochs.event_id), 4), figsize=(14, 3.5),
                             facecolor='white', constrained_layout=True)
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax, (cond, _) in zip(axes, list(epochs.event_id.items())[:4]):
        power = tfr_multitaper(epochs[cond], freqs=freqs, n_cycles=n_cycles,
                               return_itc=False, average=True, verbose=False)
        avg_power = power.data[:3].mean(axis=0)
        im = ax.imshow(avg_power, aspect='auto', origin='lower',
                       extent=[epochs.tmin, epochs.tmax, freqs[0], freqs[-1]],
                       cmap='RdBu_r',
                       vmin=np.percentile(avg_power, 5), vmax=np.percentile(avg_power, 95))
        ax.axvline(0, color='k', lw=1, ls='--')
        ax.set_title(f'ERDS — {cond}', fontsize=10)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig_path = FIGURES_DIR / f'{slug}_main.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight'); plt.close(fig)
    return {'mean_accuracy': 0.0, 'std_accuracy': 0.0,
            'figure_path': f'/figures/{slug}_main.png', 'extra_md': ''}


def run_riemann_mdm(epochs, slug: str) -> dict:
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    X = epochs.get_data()
    y = epochs.events[:, 2]
    classes, counts = np.unique(y, return_counts=True)
    top2 = classes[np.argsort(counts)[-2:]]
    mask = np.isin(y, top2)
    X, y = X[mask], y[mask]

    pipeline = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('mdm', MDM(metric=dict(mean='riemann', distance='riemann'))),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white')
    ax.bar([f'Fold {i+1}' for i in range(len(scores))], scores,
           color=['#8b5cf6' if s >= 0.7 else '#c4b5fd' for s in scores], edgecolor='white')
    ax.axhline(0.5, color='#ef4444', lw=1.2, ls='--', label='Chance')
    ax.axhline(scores.mean(), color='#6d28d9', lw=1.2, ls=':', label=f'Mean={scores.mean():.2%}')
    ax.set_ylim(0, 1); ax.set_ylabel('Accuracy')
    ax.set_title('Riemannian MDM — 5-Fold Cross-Validation')
    ax.legend(fontsize=8); fig.tight_layout()
    fig_path = FIGURES_DIR / f'{slug}_main.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight'); plt.close(fig)

    return {
        'mean_accuracy': float(scores.mean()),
        'std_accuracy': float(scores.std()),
        'figure_path': f'/figures/{slug}_main.png',
        'extra_md': f'| Fold | Accuracy |\n|------|----------|\n' +
                    '\n'.join(f'| {i+1} | {s:.1%} |' for i, s in enumerate(scores)),
    }


def run_fbcsp(epochs, slug: str) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP
    import mne

    X_raw = epochs.get_data()
    y = epochs.events[:, 2]
    classes, counts = np.unique(y, return_counts=True)
    top2 = classes[np.argsort(counts)[-2:]]
    mask = np.isin(y, top2)
    X_raw, y = X_raw[mask], y[mask]

    # Filter banks: delta, theta, alpha, beta, low-gamma
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    band_names = ['δ (1-4)', 'θ (4-8)', 'α (8-13)', 'β (13-30)', 'γ (30-40)']
    sfreq = epochs.info['sfreq']

    band_scores = []
    all_features = []
    for fmin, fmax in bands:
        # Filter epochs data for this band
        X_band = mne.filter.filter_data(X_raw, sfreq, fmin, fmax, verbose=False)
        # CSP features
        csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
        try:
            feats = []
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            pipeline = Pipeline([('csp', csp), ('lda', LinearDiscriminantAnalysis())])
            s = cross_val_score(pipeline, X_band, y, cv=cv, scoring='accuracy')
            band_scores.append(float(s.mean()))
        except Exception:
            band_scores.append(0.5)

    best_band_idx = int(np.argmax(band_scores))
    mean_acc = float(np.mean(band_scores))

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor='white')
    colors = ['#f59e0b' if i == best_band_idx else '#fde68a' for i in range(len(bands))]
    ax.bar(band_names, band_scores, color=colors, edgecolor='white')
    ax.axhline(0.5, color='#ef4444', lw=1.2, ls='--', label='Chance')
    ax.set_ylim(0, 1); ax.set_ylabel('CSP+LDA Accuracy per Band')
    ax.set_title('Filter Bank CSP — Per-Frequency-Band Decoding')
    ax.legend(fontsize=8); fig.tight_layout()
    fig_path = FIGURES_DIR / f'{slug}_main.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight'); plt.close(fig)

    extra = '| Frequency Band | Accuracy |\n|---------------|----------|\n'
    extra += '\n'.join(f'| {band_names[i]} {"⭐ best" if i == best_band_idx else ""} | {s:.1%} |'
                       for i, s in enumerate(band_scores))
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': float(np.std(band_scores)),
        'figure_path': f'/figures/{slug}_main.png',
        'extra_md': extra,
    }


def run_tangent_space_svm(epochs, slug: str) -> dict:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    X = epochs.get_data()
    y = epochs.events[:, 2]
    classes, counts = np.unique(y, return_counts=True)
    top2 = classes[np.argsort(counts)[-2:]]
    mask = np.isin(y, top2)
    X, y = X[mask], y[mask]

    pipeline = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('ts',  TangentSpace(metric='riemann')),
        ('sc',  StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale')),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white')
    ax.bar([f'Fold {i+1}' for i in range(len(scores))], scores,
           color=['#0891b2' if s >= 0.7 else '#a5f3fc' for s in scores], edgecolor='white')
    ax.axhline(0.5, color='#ef4444', lw=1.2, ls='--', label='Chance')
    ax.axhline(scores.mean(), color='#0e7490', lw=1.2, ls=':', label=f'Mean={scores.mean():.2%}')
    ax.set_ylim(0, 1); ax.set_ylabel('Accuracy')
    ax.set_title('Tangent Space + SVM — 5-Fold Cross-Validation')
    ax.legend(fontsize=8); fig.tight_layout()
    fig_path = FIGURES_DIR / f'{slug}_main.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight'); plt.close(fig)

    return {
        'mean_accuracy': float(scores.mean()),
        'std_accuracy': float(scores.std()),
        'figure_path': f'/figures/{slug}_main.png',
        'extra_md': f'| Fold | Accuracy |\n|------|----------|\n' +
                    '\n'.join(f'| {i+1} | {s:.1%} |' for i, s in enumerate(scores)),
    }


def run_shrinkage_lda(epochs, slug: str) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    X = epochs.get_data()
    y = epochs.events[:, 2]
    classes, counts = np.unique(y, return_counts=True)
    top2 = classes[np.argsort(counts)[-2:]]
    mask = np.isin(y, top2)
    X, y = X[mask], y[mask]

    # Compare standard LDA vs shrinkage LDA
    results = {}
    for solver, shrink, label in [
        ('svd',    None,         'Standard LDA'),
        ('lsqr',   'auto',       'Shrinkage LDA (Ledoit-Wolf)'),
    ]:
        pipeline = Pipeline([
            ('csp', CSP(n_components=4, reg='ledoit_wolf', log=True)),
            ('lda', LinearDiscriminantAnalysis(solver=solver, shrinkage=shrink)),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        s = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        results[label] = s

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor='white')
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w/2, results['Standard LDA'], w, label='Standard LDA', color='#94a3b8')
    ax.bar(x + w/2, results['Shrinkage LDA (Ledoit-Wolf)'], w, label='Shrinkage LDA', color='#3b82f6')
    ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.axhline(0.5, color='#ef4444', lw=1, ls='--', label='Chance')
    ax.set_ylim(0, 1); ax.set_ylabel('Accuracy')
    ax.set_title('Standard vs Shrinkage LDA — CSP Features')
    ax.legend(fontsize=8); fig.tight_layout()
    fig_path = FIGURES_DIR / f'{slug}_main.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight'); plt.close(fig)

    mean_std = float(results['Standard LDA'].mean())
    mean_shr = float(results['Shrinkage LDA (Ledoit-Wolf)'].mean())
    extra = (f'| Classifier | Mean Accuracy |\n|------------|---------------|\n'
             f'| Standard LDA | {mean_std:.1%} |\n'
             f'| Shrinkage LDA | {mean_shr:.1%} |\n'
             f'| Improvement | {(mean_shr - mean_std)*100:+.1f}pp |')
    return {
        'mean_accuracy': mean_shr,
        'std_accuracy': float(results['Shrinkage LDA (Ledoit-Wolf)'].std()),
        'figure_path': f'/figures/{slug}_main.png',
        'extra_md': extra,
    }


def run_xdawn_riemannian(epochs, slug: str) -> dict:
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    X = epochs.get_data()
    y = epochs.events[:, 2]
    classes, counts = np.unique(y, return_counts=True)
    top2 = classes[np.argsort(counts)[-2:]]
    mask = np.isin(y, top2)
    X, y = X[mask], y[mask]

    pipeline = Pipeline([
        ('xdawn_cov', XdawnCovariances(nfilter=3, estimator='oas')),
        ('ts',        TangentSpace(metric='riemann')),
        ('lr',        LogisticRegression(max_iter=500)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white')
    ax.bar([f'Fold {i+1}' for i in range(len(scores))], scores,
           color=['#059669' if s >= 0.7 else '#6ee7b7' for s in scores], edgecolor='white')
    ax.axhline(0.5, color='#ef4444', lw=1.2, ls='--', label='Chance')
    ax.axhline(scores.mean(), color='#065f46', lw=1.2, ls=':', label=f'Mean={scores.mean():.2%}')
    ax.set_ylim(0, 1); ax.set_ylabel('Accuracy')
    ax.set_title('xDAWN + Riemannian Tangent Space — 5-Fold CV')
    ax.legend(fontsize=8); fig.tight_layout()
    fig_path = FIGURES_DIR / f'{slug}_main.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight'); plt.close(fig)

    return {
        'mean_accuracy': float(scores.mean()),
        'std_accuracy': float(scores.std()),
        'figure_path': f'/figures/{slug}_main.png',
        'extra_md': f'| Fold | Accuracy |\n|------|----------|\n' +
                    '\n'.join(f'| {i+1} | {s:.1%} |' for i, s in enumerate(scores)),
    }


ALGO_RUNNERS = {
    'csp_lda':            run_csp_lda,
    'erp':                run_erp,
    'erds':               run_erds,
    'riemann_mdm':        run_riemann_mdm,
    'fbcsp':              run_fbcsp,
    'tangent_space_svm':  run_tangent_space_svm,
    'shrinkage_lda':      run_shrinkage_lda,
    'xdawn_riemannian':   run_xdawn_riemannian,
}



# ── Jupyter notebook generation ────────────────────────────────────────────────

def markdown_to_notebook(md_text: str, title: str) -> dict:
    """
    Convert a markdown post into a Jupyter .ipynb dict.
    Fenced code blocks → code cells.
    Everything else → markdown cells.
    """
    import re, json
    cells = []
    fence_re = re.compile(r'^```(\w*)\n([\s\S]*?)^```', re.MULTILINE)
    last = 0
    for m in fence_re.finditer(md_text):
        before = md_text[last:m.start()].strip()
        if before:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": before.splitlines(keepends=True),
            })
        lang = m.group(1) or "python"
        src  = m.group(2)
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.splitlines(keepends=True),
        })
        last = m.end() + 1
    after = md_text[last:].strip()
    if after:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": after.splitlines(keepends=True),
        })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "colab": {
                "name": title,
                "provenance": []
            }
        },
        "cells": cells,
    }


def write_notebook(post_path: Path, slug: str) -> None:
    """Generate a .ipynb file from a markdown post and save to notebooks/."""
    notebooks_dir = ROOT / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    text = post_path.read_text()
    # Strip YAML frontmatter
    md = text.split("---\n", 2)[-1] if text.startswith("---") else text
    nb  = markdown_to_notebook(md, slug)
    out_path = notebooks_dir / f"{slug}.ipynb"
    out_path.write_text(__import__('json').dumps(nb, indent=1, ensure_ascii=False))
    print(f"Notebook written: {out_path}")


# ── Post writer ───────────────────────────────────────────────────────────────
def write_post(meta: dict, algo: dict, result: dict,
               openai_key: str | None, date_str: str) -> Path:
    mean_acc = result['mean_accuracy']
    discussion = generate_discussion(meta, algo, mean_acc, openai_key)

    slug = f"{meta['slug']}-{algo['id'].replace('_', '-')}"
    acc_line = f'**Accuracy:** {mean_acc:.1%}  \n' if mean_acc > 0 else ''

    md = f"""---
title: "{meta['name']} — {algo['name']}"
date: "{date_str}"
dataset: "{meta['id']}"
paradigm: "{meta['paradigm']}"
subjects: 1
mean_accuracy: {mean_acc:.3f}
algorithm: "{algo['id']}"
slug: "{slug}"
---

# {meta['name']} — {algo['name']}

**Dataset:** {meta['id']} · **Algorithm:** {algo['name']} ({algo['year']})  
**Analysis date:** {date_str}
{acc_line}
---

## Algorithm Overview

{algo['desc']}

---

## Results

![Analysis result]({result['figure_path']})

{result.get('extra_md', '')}

---

## Discussion

{discussion}

---

*This report was automatically generated by the EEG Insights pipeline.*  
*Dataset: [MOABB](https://moabb.neurotechx.com) · Library: MNE-Python, scikit-learn, pyRiemann*
"""

    post_path = POSTS_DIR / f'{date_str}-{slug}.md'
    post_path.write_text(md)
    print(f'Post written: {post_path}')
    return post_path


def generate_discussion(meta: dict, algo: dict, mean_acc: float,
                        openai_key: str | None) -> str:
    import os
    api_key  = openai_key or os.environ.get('MOONSHOT_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.moonshot.ai/v1')

    if not api_key:
        perf = 'above chance' if mean_acc > 0.6 else 'near chance'
        return (
            f"The **{algo['name']}** algorithm applied to {meta['name']} yields {perf} performance "
            f"({mean_acc:.1%}). {algo['desc']} "
            f"Future work could compare this against deep learning baselines such as EEGNet or ShallowConvNet."
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = (
        f"Write a concise 3-paragraph scientific discussion for an EEG analysis blog post.\n"
        f"Dataset: {meta['name']} (motor imagery BCI).\n"
        f"Algorithm: {algo['name']} — {algo['desc']}\n"
        f"Mean accuracy: {mean_acc:.1%}.\n"
        f"Cover: (1) interpretation of the result for BCI research, "
        f"(2) specific strengths and limitations of {algo['name']}, "
        f"(3) future directions or comparisons with other methods. "
        f"Use neuroscience/BCI terminology. ~180 words. No headers."
    )
    try:
        resp = client.chat.completions.create(
            model='moonshot-v1-8k',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f'Kimi API error: {e}')
        return (
            f"The **{algo['name']}** algorithm applied to {meta['name']} yields {mean_acc:.1%} accuracy. "
            f"{algo['desc']}"
        )


# ── Post index ────────────────────────────────────────────────────────────────
def update_index():
    posts = []
    for f in sorted(POSTS_DIR.glob('*.md'), reverse=True):
        text = f.read_text()
        lines = text.split('\n')
        fm = {}
        if lines[0] == '---':
            for line in lines[1:]:
                if line == '---':
                    break
                if ':' in line:
                    k, v = line.split(':', 1)
                    fm[k.strip()] = v.strip().strip('"')
        posts.append({
            'file': f.name,
            'slug': fm.get('slug', f.stem),
            'title': fm.get('title', f.stem),
            'date': fm.get('date', ''),
            'dataset': fm.get('dataset', ''),
            'paradigm': fm.get('paradigm', ''),
            'subjects': fm.get('subjects', ''),
            'mean_accuracy': fm.get('mean_accuracy', ''),
            'algorithm': fm.get('algorithm', ''),
        })
    index_path = ROOT / 'frontend' / 'public' / 'posts-index.json'
    index_path.write_text(json.dumps(posts, indent=2))
    print(f'Index updated: {len(posts)} posts')


# ── Custom analysis loader ────────────────────────────────────────────────────
def run_custom(epochs, slug, custom_name):
    import types
    custom_path = ROOT / 'pipeline' / 'custom_analyses.json'
    if not custom_path.exists():
        raise FileNotFoundError('custom_analyses.json not found')
    analyses = json.loads(custom_path.read_text())
    if custom_name not in analyses:
        raise KeyError(f'Custom analysis "{custom_name}" not found')
    code = analyses[custom_name]['code']
    mod = types.ModuleType('custom_mod')
    exec(compile(code, '<custom>', 'exec'), mod.__dict__)
    if not hasattr(mod, 'custom_analysis'):
        raise AttributeError('Function custom_analysis() not found in code')
    return mod.custom_analysis(epochs, slug, FIGURES_DIR)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    default=None)
    parser.add_argument('--subject',    type=int, default=1)
    parser.add_argument('--all-subjects', action='store_true')
    parser.add_argument('--auto',       action='store_true')
    parser.add_argument('--algo',       default=None, help='Force algorithm id')
    parser.add_argument('--custom',     default=None, help='Run saved custom analysis')
    parser.add_argument('--rerun-slug', default=None)
    args = parser.parse_args()

    openai_key = os.environ.get('OPENAI_API_KEY')
    date_str   = datetime.now().strftime('%Y-%m-%d')
    state      = load_state()

    # Pick dataset
    if args.dataset:
        meta_raw = next((d for d in DATASETS if d['id'] == args.dataset), DATASETS[0])
    else:
        meta_raw = next_dataset(state)
    meta = {**meta_raw, 'slug': meta_raw['id'].lower().replace('_', '-')}
    print(f'Dataset: {meta["name"]} ({meta["id"]})')

    # Pick algorithm
    if args.algo:
        algo = next((a for a in ALGORITHMS if a['id'] == args.algo), ALGORITHMS[0])
    else:
        algo = next_algo(state)
    print(f'Algorithm: {algo["name"]} ({algo["year"]})')

    # Load data
    print(f'  Subject {args.subject}...')
    _, sessions = load_dataset(meta['id'], args.subject)
    epochs = get_epochs(sessions, args.subject)

    slug = f"{meta['slug']}-{algo['id'].replace('_', '-')}"

    # Run algorithm
    if args.custom:
        print(f'  Running custom analysis: {args.custom}')
        result = run_custom(epochs, slug, args.custom)
    elif algo['id'] in ALGO_RUNNERS:
        result = ALGO_RUNNERS[algo['id']](epochs, slug)
    else:
        print(f'Unknown algorithm {algo["id"]}, falling back to CSP+LDA')
        result = run_csp_lda(epochs, slug)

    acc = result.get('mean_accuracy', 0)
    if acc > 0:
        print(f'  Accuracy: {acc:.1%} ± {result.get("std_accuracy", 0):.1%}')

    # Write post + update index
    print('Writing post...')
    post_file = write_post(meta, algo, result, openai_key, date_str)
    update_index()
    write_notebook(post_file, slug)
    save_state(state)
    print('Done.')


if __name__ == '__main__':
    main()
