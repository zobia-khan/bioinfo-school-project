# 🔥 SSAQS Burnout Risk Tracker

A data analysis project that computes a daily **Burnout Risk Score (0–100)** for 35 university students using the [SSAQS dataset](https://doi.org/10.xxxx/ssaqs) — combining wearable-sensor data with daily psychological questionnaires.

---

## 📋 Overview

The tracker fuses four evidence-based signal domains into a single composite score:

| Domain | Signal | Weight |
|---|---|---|
| Psychological load | Stress + Anxiety (questionnaire) | 35 % |
| Autonomic dysregulation | HRV — RMSSD (wearable) | 30 % |
| Physical deconditioning | 7-day step decline (wearable) | 20 % |
| Sleep impairment | Sleep quality score (wearable) | 15 % |

### Risk Tiers

| Score | Label |
|---|---|
| 0 – 40 | 🟢 Low |
| 40 – 65 | 🟡 Moderate |
| 65 – 100 | 🔴 High |

---

## 📁 Repository Structure

```
bioinfo-school-project/
├── burnout_tracker.py          # Batch analysis script (all 35 participants)
├── burnout_self_assessment.html# ✨ Patient-facing self-assessment tool (no Python needed)
├── burnout_report.html         # Generated cohort report (output of tracker)
├── requirements.txt            # Python dependencies
└── SSAQS dataset/
    ├── README.txt          # Original dataset description & citation
    ├── users-courses.csv   # Participant → course & university mapping
    ├── course-details.csv  # Course metadata
    └── 1/ … 35/            # Per-participant data folders
        ├── daily_questions.csv
        ├── hrv.csv
        ├── steps.csv
        └── sleep.csv
```

---

## ✨ Patient Self-Assessment Tool

[`burnout_self_assessment.html`](burnout_self_assessment.html) is a **standalone interactive tool** — open it in any browser with no Python or server required.

### Features
- **4-step guided questionnaire**: mind → body tension → activity → sleep
- **Patient-friendly inputs**: qualitative option cards or wearable data (RMSSD / step count)
- **Animated gauge** with live score count-up
- **Plain-English explanation** of what your score means
- **Score formula transparency**: shows exactly how each domain contributes to the total
- **Personalised domain insights**: what each signal means for *you* specifically
- **Actionable recommendations** tailored to your risk level

### Score formula (same logic as `burnout_tracker.py`)

```
Score = 35% × Psychological Load  +  30% × Nervous System Tension
      + 20% × Physical Activity   +  15% × Sleep Quality
```

All domain values are normalised to 0–100 before weighting (0 = healthy, 100 = most at risk).

---

## 🚀 Usage (batch analysis)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or with `uv` (recommended):

```bash
uv pip install -r requirements.txt
```

### 2. Run the tracker

```bash
# Using the default data directory (SSAQS dataset/ next to the script)
python burnout_tracker.py

# Custom paths
python burnout_tracker.py --data-dir "path/to/SSAQS dataset" --out my_report.html
```

### 3. View the report

Open the generated **`burnout_report.html`** in any web browser — it is fully self-contained (no server required).

#### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `./SSAQS dataset` | Path to the SSAQS dataset folder |
| `--out` | `./burnout_report.html` | Output HTML file path |

---

## 📊 Report Contents

The HTML report includes four interactive Plotly sections:

1. **Cohort Heatmap** — burnout scores for all 35 students across every day
2. **Course Comparison** — box plot comparing burnout distributions across course groups (A1, A2, B)
3. **Student Summary Table** — sortable overview with average score, peak score, peak day, and high-risk day count
4. **Individual Timelines** — expandable per-student view showing the burnout score curve and per-domain contributions (stress, HRV, steps, sleep)

---

## 🧬 Dataset

**A Dataset of University Students' Stress and Anxiety Levels based on Questionnaires and Wearable Sensors**

> Garcia-Ceja, E., Alvarado-Uribe, J., Escamilla-Ambrosio, P. J., Lara, A., Mena-Martinez, A., Gallegos-Garcia, G., Gonzalez-Mendoza, M., Monroy, R., Martinez Luna, G., & Fernández-Cárdenas, J. M. (2026).

- 35 university participants across two universities and three course groups (A1, A2, B)
- Per-participant folders (`1/` – `35/`) each containing CSV files for HRV, steps, sleep, and daily questionnaire responses
- Please cite the paper above if you use this dataset in your own work.

---

## ⚙️ Dependencies

| Package | Version |
|---|---|
| pandas | ≥ 2.0 |
| plotly | ≥ 5.0 |
| numpy | ≥ 1.24 |

---

## 📝 Notes

- HRV windows with `coverage < 0.7` are dropped as unreliable.
- SpO₂ sentinel value `50` (no-signal marker) is excluded.
- Missing signal days are imputed conservatively (forward-fill ≤ 2 days; neutral `0.5` fallback).
- The final burnout score is smoothed with a **7-day rolling mean** to reduce noise.
