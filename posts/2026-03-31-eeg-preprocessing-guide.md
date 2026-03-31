---
title: "EEG Signal Preprocessing: A Practical Guide"
date: "2026-03-31"
dataset: "Tutorial"
paradigm: "preprocessing"
subjects: 0
mean_accuracy: ""
slug: "eeg-preprocessing-guide"
---

# EEG Signal Preprocessing: A Practical Guide

**Type:** Educational · **Topic:** Signal Preprocessing · **Library:** MNE-Python

> Raw EEG is almost never ready to analyse. This guide walks through every standard
> preprocessing step — what it does, why it matters, and how to do it in MNE-Python.

---

## Why Preprocessing Matters

EEG records tiny electrical potentials on the scalp (typically 1–100 µV). The problem
is that many other signals are far larger:

| Artifact | Typical amplitude | Relative to EEG |
|----------|------------------|-----------------|
| Eye blink (EOG) | ~100–500 µV | 10–100× larger |
| Muscle noise (EMG) | ~50–200 µV | 5–50× larger |
| Power line noise | ~20 µV | 2–20× larger |
| Cardiac (ECG) | ~10–50 µV | 1–10× larger |
| True EEG signal | ~1–100 µV | — |

Without preprocessing, these artifacts dominate every analysis. A pipeline that skips
preprocessing will decode the participant's blinking pattern, not their brain state.

---

## 1. Loading and Inspecting Raw Data

The starting point is always the raw recording. In MNE, every dataset loads as a
`Raw` object containing the continuous signal and channel metadata.

```python
import mne

# Load from file (EDF, BrainVision, EEGLab, etc.)
raw = mne.io.read_raw_edf('subject01.edf', preload=True)

# Check what you have
print(raw.info)
print(f"Duration: {raw.times[-1]:.1f} s")
print(f"Channels: {len(raw.ch_names)}")
print(f"Sample rate: {raw.info['sfreq']} Hz")

# Quick visual inspection — always look at raw data first
raw.plot(n_channels=20, duration=10, scalings='auto')
```

**What to look for:**
- Channels with flat lines (broken electrode)
- Channels with excessive noise (poor contact)
- Obvious large artifacts (movement, sweat)
- Power line interference (50/60 Hz hum)

---

## 2. Channel Selection and Montage

EEG channels have standard spatial locations (10-20, 10-10 system). Setting the
montage is required for any spatial analysis (ICA, source localisation, topomaps).

```python
# Set standard 10-20 montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Drop non-EEG channels (EOG, EMG, triggers) before analysis
raw.pick_types(eeg=True, eog=False, emg=False, stim=False)

# Alternatively, keep EOG for artifact removal
raw.pick_types(eeg=True, eog=True)
```

**The 10-20 system:** Electrode positions are defined by percentages of skull
distances (10% or 20% of nasion-inion / preauricular distances). Key landmarks:
- **Fz, Cz, Pz** — midline frontal, central, parietal
- **C3, C4** — left/right motor cortex (critical for motor imagery)
- **O1, O2** — occipital (visual cortex)
- **Fp1, Fp2** — frontal pole (most prone to eye artifacts)

---

## 3. Re-referencing

EEG voltages are always measured relative to a reference electrode. The choice of
reference changes the spatial distribution of your signal significantly.

```python
# Common average reference (recommended for dense arrays)
raw.set_eeg_reference('average', projection=True)
raw.apply_proj()

# Linked mastoids reference (traditional, good for auditory)
raw.set_eeg_reference(['TP9', 'TP10'])

# REST (Reference Electrode Standardization Technique) — most neutral
raw.set_eeg_reference('REST')
```

| Reference | Best for | Notes |
|-----------|----------|-------|
| Common average | Dense arrays (64+ ch) | Minimises reference bias |
| Linked mastoids | Auditory, clinical ERP | Traditional standard |
| Cz | Motor imagery | Poor if Cz is your channel of interest |
| REST | General purpose | Requires realistic head model |

**Critical rule:** Never analyse data with the original reference electrode included —
it will always appear silent (zero voltage by definition).

---

## 4. Filtering

Filtering removes frequency components outside the band of interest. Two filters
are always applied:

### 4.1 High-pass filter (remove drift)

Slow drifts from sweat, electrode movement, and breathing contaminate the baseline.
A high-pass filter above ~0.1–1 Hz removes them.

```python
# High-pass at 1 Hz — removes slow drift
# Note: 0.1 Hz is safer for ERP studies to preserve slow components
raw.filter(l_freq=1.0, h_freq=None, method='fir', fir_window='hamming')
```

### 4.2 Low-pass filter (remove high-frequency noise)

Muscle noise and external interference appear at high frequencies. A low-pass
filter at 40–100 Hz removes most of it without affecting EEG bands of interest.

```python
# Low-pass at 40 Hz — removes muscle noise, keeps all EEG bands
raw.filter(l_freq=None, h_freq=40.0, method='fir', fir_window='hamming')

# Bandpass in one call (equivalent to both above)
raw.filter(l_freq=1.0, h_freq=40.0)
```

### 4.3 Notch filter (remove power line noise)

Power line interference creates a sharp peak at 50 Hz (Europe/Asia) or 60 Hz (Americas).

```python
# Remove 50 Hz and its harmonics (100, 150 Hz)
raw.notch_filter(freqs=[50, 100, 150])

# Or for 60 Hz (Americas)
raw.notch_filter(freqs=[60, 120, 180])
```

**EEG frequency bands of interest:**

| Band | Range | Associated with |
|------|-------|-----------------|
| Delta | 0.5–4 Hz | Deep sleep, pathology |
| Theta | 4–8 Hz | Drowsiness, memory encoding |
| Alpha | 8–13 Hz | Relaxation, visual suppression |
| **Mu** | **8–12 Hz** | **Motor cortex (motor imagery)** |
| Beta | 13–30 Hz | Active thinking, motor planning |
| Gamma | 30–80 Hz | High-level cognition, binding |

---

## 5. Detecting and Removing Bad Channels

A bad channel is one with a corrupted signal — broken electrode, poor impedance,
or bridged contacts. Including bad channels in analysis spreads their noise to
neighbouring channels via interpolation-based methods.

```python
# Automated bad channel detection
from mne.preprocessing import find_bad_channels_maxwell

# Visual inspection (most reliable)
raw.plot(n_channels=30, block=True)
# Mark bad channels interactively or programmatically:
raw.info['bads'] = ['F3', 'CP5']  # channels identified as bad

# Interpolate bad channels from neighbours
raw.interpolate_bads(reset_bads=True)
```

**Signs of a bad channel:**
- **Flat** — amplitude near zero for extended periods
- **Noisy** — amplitude 3–5× higher than neighbours
- **Bridged** — nearly identical to adjacent channel (electrode gel bridging)
- **Step artifacts** — sudden DC jumps

---

## 6. Independent Component Analysis (ICA)

ICA is the most powerful method for removing stereotyped artifacts — eye blinks,
horizontal eye movements, and cardiac artifacts. It decomposes the signal into
statistically independent components, identifies artifact components, and removes them.

```python
from mne.preprocessing import ICA

# Fit ICA (use 'fastica' or 'infomax')
ica = ICA(n_components=20, method='fastica', random_state=42, max_iter='auto')
ica.fit(raw, picks='eeg')

# Plot components to identify artifacts
ica.plot_components()          # topomaps — blink = frontal dipole
ica.plot_sources(raw)          # time series — blink = slow large spikes

# Automatic blink detection using EOG channel
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp1')
ica.exclude = eog_indices

# Manual exclusion after visual inspection
ica.exclude = [0, 3]   # components 0 and 3 are artifacts

# Apply — removes artifact components from signal
raw_clean = ica.apply(raw.copy())
```

**How to identify artifact components:**

| Component shape | Artifact type | Action |
|----------------|---------------|--------|
| Frontal bilateral dipole | Eye blink (EOG) | Exclude |
| Horizontal frontal | Horizontal eye movement | Exclude |
| Regular heartbeat rhythm | Cardiac (ECG) | Exclude |
| Diffuse high-frequency | Muscle (EMG) | Exclude |
| Focal scalp distribution | Brain signal | Keep |

---

## 7. Epoching

After cleaning the continuous signal, we cut it into fixed-length segments
(epochs) time-locked to stimulus events or triggers.

```python
# Find events from stimulus channel or annotations
events, event_id = mne.events_from_annotations(raw_clean)

print("Event types found:")
for name, code in event_id.items():
    count = (events[:, 2] == code).sum()
    print(f"  {name} (code {code}): {count} trials")

# Create epochs: -0.5 to 2.5 s around each event
epochs = mne.Epochs(
    raw_clean,
    events,
    event_id=event_id,
    tmin=-0.5,      # 500 ms before stimulus
    tmax=2.5,       # 2500 ms after stimulus
    baseline=(-0.5, 0),   # baseline correct using pre-stimulus period
    preload=True,
    reject=None,    # we'll do rejection below
    verbose=False
)

print(f"Epochs created: {len(epochs)}")
```

---

## 8. Epoch Rejection

Even after ICA, some epochs contain non-stationary artifacts too large or irregular
to model (head movement, electrode pop). We reject epochs exceeding amplitude thresholds.

```python
# Peak-to-peak amplitude rejection threshold
reject_criteria = {
    'eeg': 150e-6,   # 150 µV — epochs with larger swings are discarded
}

epochs.drop_bad(reject=reject_criteria)
print(f"Epochs after rejection: {len(epochs)}")
print(f"Dropped: {epochs.drop_log}")

# Auto-rejection with adaptive threshold (more sophisticated)
from autoreject import AutoReject
ar = AutoReject(random_state=42)
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
reject_log.plot()
```

**Rejection guidelines:**
- For motor imagery: 100–200 µV is typical
- For high-density arrays: 80–120 µV
- Aim to keep at least 75% of trials; if dropping more, check your preprocessing

---

## 9. Baseline Correction

Baseline correction removes pre-stimulus activity differences between trials,
normalising each epoch relative to a stable reference window before the event.

```python
# Apply baseline correction (if not done at epoching)
# Subtracts mean of baseline window from entire epoch
epochs.apply_baseline((-0.5, 0))

# For time-frequency: percent change from baseline
from mne.baseline import rescale
# (applied per-frequency during tfr computation)
```

---

## 10. Complete Preprocessing Pipeline

Here is a full, production-ready pipeline combining all steps:

```python
import mne
from mne.preprocessing import ICA

def preprocess_eeg(raw_path: str, sfreq_resample: float = 250) -> mne.Epochs:
    """
    Full EEG preprocessing pipeline.
    Returns clean epochs ready for analysis.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────
    raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
    print(f"Loaded: {len(raw.ch_names)} channels, {raw.times[-1]:.0f} s")

    # ── 2. Channel setup ─────────────────────────────────────────────────
    raw.pick_types(eeg=True, eog=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # ── 3. Resample (optional — speeds up ICA) ───────────────────────────
    if raw.info['sfreq'] > sfreq_resample:
        raw.resample(sfreq_resample)

    # ── 4. Filter ────────────────────────────────────────────────────────
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    raw.notch_filter(freqs=[50, 100], verbose=False)

    # ── 5. Re-reference ──────────────────────────────────────────────────
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()

    # ── 6. Detect and interpolate bad channels ───────────────────────────
    raw.info['bads'] = []
    # (add automated detection here if desired)
    raw.interpolate_bads(reset_bads=True)

    # ── 7. ICA artifact removal ───────────────────────────────────────────
    ica = ICA(n_components=15, method='fastica', random_state=42, max_iter='auto')
    ica.fit(raw, picks='eeg', verbose=False)
    eog_idx, _ = ica.find_bads_eog(raw, ch_name='Fp1', threshold=3.0)
    ica.exclude = eog_idx
    raw = ica.apply(raw)

    # ── 8. Epoch ─────────────────────────────────────────────────────────
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=-0.5, tmax=2.5,
        baseline=(-0.5, 0),
        preload=True,
        reject={'eeg': 150e-6},
        verbose=False,
    )

    # ── 9. Report ─────────────────────────────────────────────────────────
    n_dropped = len(epochs.drop_log) - len(epochs)
    print(f"Clean epochs: {len(epochs)} ({n_dropped} dropped, "
          f"{n_dropped/len(epochs.drop_log)*100:.0f}% rejection rate)")

    return epochs
```

---

## Common Preprocessing Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Filtering after epoching | Edge artifacts at epoch boundaries | Always filter on continuous data |
| High-pass > 1 Hz for ERPs | Distorts slow ERP components (P300) | Use 0.1 Hz for ERP, 1 Hz for BCI |
| Skipping bad channel interpolation | CSP/ICA corrupted by flat channels | Always interpolate before ICA |
| Over-aggressive ICA exclusion | Removes brain signal | Exclude ≤ 5 components; verify topomaps |
| Too strict rejection threshold | Too few trials for reliable analysis | Keep at least 40 trials per class |
| Wrong notch frequency | Power line hum remains | 50 Hz (Europe), 60 Hz (Americas) |

---

## Summary

A typical EEG preprocessing pipeline follows this order:

```
Raw signal
  ↓ Channel setup + montage
  ↓ Resample (optional)
  ↓ Bandpass filter (1–40 Hz)
  ↓ Notch filter (50 or 60 Hz)
  ↓ Re-reference (average)
  ↓ Bad channel interpolation
  ↓ ICA (remove blinks + cardiac)
  ↓ Epoching (event-locked segments)
  ↓ Baseline correction
  ↓ Epoch rejection (amplitude threshold)
  ↓ Clean epochs → ready for analysis
```

Each step builds on the previous. Skipping any step risks contaminating your
downstream analysis with artifacts that are often far larger than the neural
signal you care about.

---

*Written for the EEG Insights pipeline · Library references: [MNE-Python](https://mne.tools) · [autoreject](https://autoreject.github.io)*
