"""
burnout_tracker.py
==================
Burnout Risk Tracker built on the SSAQS dataset.

For each of the 35 participants it computes a daily Burnout Risk Score (0–100)
from four evidence-based domains:

  Domain                  Signal              Weight
  ─────────────────────── ─────────────────── ──────
  Psychological load      stress + anxiety    35 %
  Autonomic dysregulation HRV (RMSSD)         30 %
  Physical deconditioning 7-day step decline  20 %
  Sleep impairment        sleep quality score 15 %

Output: burnout_report.html  (self-contained, no server needed)

Usage:
    python burnout_tracker.py
    python burnout_tracker.py --data-dir "path/to/SSAQS dataset" --out burnout_report.html
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

WEIGHTS = dict(stress=0.35, hrv=0.30, steps=0.20, sleep=0.15)

# RMSSD ≥ 80 ms  → healthy (score 0); RMSSD ≤ 20 ms → very stressed (score 1)
HRV_HEALTHY_RMSSD = 80.0
HRV_LOW_RMSSD     = 20.0

# SpO₂ sentinel value — device reports 50 when it has no signal
SPO2_SENTINEL = 50.0

# Step baseline window (days from start) used to compute "normal" activity
STEPS_BASELINE_DAYS = 14

# Rolling smoothing window for the final score
SMOOTH_DAYS = 7

RISK_COLORS = {
    "Low":      "#2ecc71",
    "Moderate": "#f39c12",
    "High":     "#e74c3c",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _to_utc(s: pd.Series) -> pd.Series:
    """Safely coerce a datetime Series to UTC-aware, regardless of current state."""
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def load_daily_questions(path: Path) -> pd.DataFrame:
    """Returns daily rows with columns: date, stress_norm (0-1)."""
    df = pd.read_csv(path)
    # timeStampStart is Unix epoch in seconds
    df["date"] = _to_utc(pd.to_datetime(df["timeStampStart"], unit="s").dt.normalize())
    df["stress_norm"] = (df["stress"].clip(0, 100) + df["anxiety"].clip(0, 100)) / 2 / 100
    return df.groupby("date", as_index=False)["stress_norm"].mean()


def load_hrv(path: Path) -> pd.DataFrame:
    """Returns daily mean RMSSD normalised to 0-1 burnout risk."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = _to_utc(df["timestamp"].dt.normalize())
    # Coverage < 0.7 → unreliable; drop those windows
    df = df[df["coverage"] >= 0.7]
    daily = df.groupby("date", as_index=False)["rmssd"].mean()
    # Low RMSSD = high risk → invert and normalise
    daily["hrv_norm"] = 1.0 - (
        (daily["rmssd"].clip(HRV_LOW_RMSSD, HRV_HEALTHY_RMSSD) - HRV_LOW_RMSSD)
        / (HRV_HEALTHY_RMSSD - HRV_LOW_RMSSD)
    )
    return daily[["date", "hrv_norm", "rmssd"]]


def load_steps(path: Path, baseline_days: int = STEPS_BASELINE_DAYS) -> pd.DataFrame:
    """Returns daily total steps and a drop-from-baseline normalised score."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = _to_utc(df["timestamp"].dt.normalize())
    daily = df.groupby("date", as_index=False)["steps"].sum()
    daily = daily.sort_values("date").reset_index(drop=True)

    # Baseline = mean steps over first `baseline_days` days
    baseline = daily.head(baseline_days)["steps"].mean()
    if baseline == 0 or np.isnan(baseline):
        baseline = daily["steps"].mean()
    if baseline == 0:
        daily["steps_norm"] = 0.0
    else:
        # 7-day rolling average then compare to baseline
        daily["steps_7d"] = daily["steps"].rolling(7, min_periods=1).mean()
        daily["steps_norm"] = (1.0 - daily["steps_7d"] / baseline).clip(0, 1)

    return daily[["date", "steps", "steps_norm"]]


def load_sleep(path: Path) -> pd.DataFrame:
    """Returns daily sleep impairment score (0-1; 1 = worst sleep)."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = _to_utc(df["timestamp"].dt.normalize())
    daily = df.groupby("date", as_index=False).agg(
        sleep_score=("overall_score", "mean"),
        deep_sleep_min=("deep_sleep_in_minutes", "mean"),
    )
    # Lower score = worse sleep = higher burnout risk
    daily["sleep_norm"] = 1.0 - (daily["sleep_score"].clip(0, 100) / 100.0)
    return daily[["date", "sleep_score", "deep_sleep_min", "sleep_norm"]]


# ──────────────────────────────────────────────────────────────────────────────
# Composite score
# ──────────────────────────────────────────────────────────────────────────────

def compute_burnout(uid: int, user_dir: Path) -> pd.DataFrame | None:
    """Load all signals for one user and return a daily burnout DataFrame."""
    files = {
        "q":     user_dir / "daily_questions.csv",
        "hrv":   user_dir / "hrv.csv",
        "steps": user_dir / "steps.csv",
        "sleep": user_dir / "sleep.csv",
    }
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        print(f"  [WARN] User {uid}: missing files {missing} — skipping")
        return None

    try:
        q  = load_daily_questions(files["q"])
        h  = load_hrv(files["hrv"])
        s  = load_steps(files["steps"])
        sl = load_sleep(files["sleep"])
    except Exception as exc:
        print(f"  [WARN] User {uid}: error loading data — {exc}")
        return None

    # Ensure all dates are UTC-aware for merging
    for df in (q, h, s, sl):
        df["date"] = _to_utc(df["date"])

    # Merge on date (outer so we keep all days)
    merged = q.merge(h, on="date", how="outer")
    merged = merged.merge(s, on="date", how="outer")
    merged = merged.merge(sl, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Forward-fill sleep (daily) and back-fill HRV gaps ≤ 2 days
    merged["sleep_norm"]  = merged["sleep_norm"].ffill(limit=2).fillna(0.5)
    merged["sleep_score"] = merged["sleep_score"].ffill(limit=2)
    merged["hrv_norm"]    = merged["hrv_norm"].interpolate(limit=2).fillna(0.5)
    merged["rmssd"]       = merged["rmssd"].interpolate(limit=2)
    merged["steps_norm"]  = merged["steps_norm"].fillna(0.0)
    merged["stress_norm"] = merged["stress_norm"].fillna(np.nan)  # keep NaN if no survey

    # Composite score (0–100)
    merged["burnout_raw"] = (
        WEIGHTS["stress"] * merged["stress_norm"].fillna(0.5)
        + WEIGHTS["hrv"]    * merged["hrv_norm"]
        + WEIGHTS["steps"]  * merged["steps_norm"]
        + WEIGHTS["sleep"]  * merged["sleep_norm"]
    ) * 100.0

    # Smooth with rolling 7-day window
    merged["burnout_score"] = (
        merged["burnout_raw"].rolling(SMOOTH_DAYS, min_periods=1).mean()
    )

    # Risk label
    merged["risk"] = pd.cut(
        merged["burnout_score"],
        bins=[-1, 40, 65, 101],
        labels=["Low", "Moderate", "High"],
    ).astype(str)

    merged["user_id"] = uid
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# HTML report builder
# ──────────────────────────────────────────────────────────────────────────────

def risk_color_list(risks: pd.Series) -> list[str]:
    return [RISK_COLORS.get(r, "#95a5a6") for r in risks]


def build_report(all_data: dict[int, pd.DataFrame], users_courses: pd.DataFrame, out: Path):
    """Build a single self-contained HTML report with Plotly figures."""

    user_ids  = sorted(all_data.keys())
    n_users   = len(user_ids)

    # ── 1. Cohort heatmap ─────────────────────────────────────────────────────
    all_dates = sorted(
        set(d for df in all_data.values() for d in df["date"].dt.date.tolist())
    )
    heat_z = []
    heat_y = []
    for uid in user_ids:
        df = all_data[uid].copy()
        df["day"] = df["date"].dt.date
        pivot = df.set_index("day")["burnout_score"].reindex(all_dates)
        heat_z.append(pivot.tolist())
        heat_y.append(f"User {uid}")

    heat_fig = go.Figure(go.Heatmap(
        z=heat_z,
        x=[str(d) for d in all_dates],
        y=heat_y,
        colorscale=[
            [0.0,  "#2ecc71"],
            [0.40, "#f1c40f"],
            [0.65, "#e74c3c"],
            [1.0,  "#8e0000"],
        ],
        zmin=0, zmax=100,
        colorbar=dict(title="Burnout Risk Score"),
        hovertemplate="Date: %{x}<br>User: %{y}<br>Score: %{z:.1f}<extra></extra>",
    ))
    heat_fig.update_layout(
        title="🌡️ Cohort Burnout Heatmap — All Students × All Days",
        xaxis_title="Date",
        yaxis_title="Participant",
        height=max(400, n_users * 22),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font_color="#e0e0e0",
        xaxis=dict(tickangle=-45, nticks=20),
    )

    # ── 2. Summary table ──────────────────────────────────────────────────────
    summary_rows = []
    for uid in user_ids:
        df = all_data[uid]
        course_row = users_courses[users_courses["userid"] == uid]
        course = course_row["course"].values[0] if not course_row.empty else "?"
        univ   = course_row["university"].values[0] if not course_row.empty else "?"

        peak_idx  = df["burnout_score"].idxmax()
        peak_day  = df.loc[peak_idx, "date"].date() if not df.empty else None
        peak_val  = df["burnout_score"].max()
        avg_val   = df["burnout_score"].mean()
        high_days = (df["risk"] == "High").sum()
        summary_rows.append(dict(
            user_id=uid,
            university=univ,
            course=course,
            avg_score=round(avg_val, 1),
            peak_score=round(peak_val, 1),
            peak_day=str(peak_day),
            high_risk_days=int(high_days),
        ))

    summary_df = pd.DataFrame(summary_rows).sort_values("avg_score", ascending=False)

    table_fig = go.Figure(go.Table(
        header=dict(
            values=["User", "Univ.", "Course", "Avg Score", "Peak Score", "Peak Day", "High-Risk Days"],
            fill_color="#0f3460",
            font=dict(color="white", size=13),
            align="center",
        ),
        cells=dict(
            values=[
                summary_df["user_id"],
                summary_df["university"],
                summary_df["course"],
                summary_df["avg_score"],
                summary_df["peak_score"],
                summary_df["peak_day"],
                summary_df["high_risk_days"],
            ],
            fill_color=[
                ["#1a1a2e"] * len(summary_df),
                ["#1a1a2e"] * len(summary_df),
                ["#1a1a2e"] * len(summary_df),
                [
                    "#8e0000" if v >= 65 else "#b7770d" if v >= 40 else "#1a6b3a"
                    for v in summary_df["avg_score"]
                ],
                ["#1a1a2e"] * len(summary_df),
                ["#1a1a2e"] * len(summary_df),
                ["#1a1a2e"] * len(summary_df),
            ],
            font=dict(color="white", size=12),
            align="center",
        ),
    ))
    table_fig.update_layout(
        title="📋 Student Summary Table",
        paper_bgcolor="#16213e",
        font_color="#e0e0e0",
        height=max(300, len(summary_df) * 28 + 80),
        margin=dict(t=60, b=10),
    )

    # ── 3. Individual timelines ───────────────────────────────────────────────
    individual_figs = []
    for uid in user_ids:
        df = all_data[uid].copy()
        df["day_str"] = df["date"].dt.strftime("%Y-%m-%d")
        course_row = users_courses[users_courses["userid"] == uid]
        course = course_row["course"].values[0] if not course_row.empty else "?"

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.08,
            subplot_titles=("Burnout Risk Score", "Domain Contributions"),
        )

        # Risk band shading
        for lo, hi, label, col in [
            (0, 40, "Low", "rgba(46,204,113,0.08)"),
            (40, 65, "Moderate", "rgba(243,156,18,0.10)"),
            (65, 100, "High", "rgba(231,76,60,0.12)"),
        ]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=col, layer="below", line_width=0, row=1, col=1)

        # Burnout score line
        fig.add_trace(go.Scatter(
            x=df["day_str"], y=df["burnout_score"],
            mode="lines",
            name="Burnout Score",
            line=dict(color="#e056fd", width=2.5),
            hovertemplate="%{x}<br>Score: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

        # Survey stress markers (only on days a questionnaire was completed)
        has_survey = df["stress_norm"].notna()
        if has_survey.any():
            fig.add_trace(go.Scatter(
                x=df.loc[has_survey, "day_str"],
                y=df.loc[has_survey, "burnout_raw"],
                mode="markers",
                name="Survey day",
                marker=dict(color="#fdcb6e", size=5, symbol="circle-open"),
                hovertemplate="%{x}<br>Raw score: %{y:.1f}<extra></extra>",
            ), row=1, col=1)

        # Domain stacked area
        domain_cols = {
            "Stress / Anxiety": ("stress_norm", "rgba(225,112,85,0.7)"),
            "HRV (autonomic)":  ("hrv_norm",    "rgba(116,185,255,0.7)"),
            "Step decline":     ("steps_norm",  "rgba(85,239,196,0.7)"),
            "Sleep quality":    ("sleep_norm",  "rgba(162,155,254,0.7)"),
        }
        for label, (col, color) in domain_cols.items():
            vals = df[col].fillna(0) * 100
            fig.add_trace(go.Scatter(
                x=df["day_str"], y=vals,
                mode="lines",
                name=label,
                stackgroup="domains",
                line=dict(color=color, width=0.5),
                fillcolor=color,
                hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>",
            ), row=2, col=1)

        fig.update_layout(
            title=f"User {uid} — Course {course}",
            height=520,
            plot_bgcolor="#1a1a2e",
            paper_bgcolor="#16213e",
            font_color="#e0e0e0",
            legend=dict(orientation="h", y=-0.15, font_size=11),
            hovermode="x unified",
        )
        fig.update_yaxes(range=[0, 100], title_text="Score", row=1, col=1)
        fig.update_yaxes(range=[0, 100], title_text="Contribution (%)", row=2, col=1)
        fig.update_xaxes(tickangle=-30)

        individual_figs.append((uid, course, fig))

    # ── 4. Course comparison box plot ─────────────────────────────────────────
    all_df = pd.concat(all_data.values(), ignore_index=True)
    all_df = all_df.rename(columns={"user_id": "userid"})
    all_df = all_df.merge(users_courses[["userid", "course", "university"]], on="userid", how="left")

    box_fig = px.box(
        all_df.dropna(subset=["burnout_score"]),
        x="course", y="burnout_score",
        color="course",
        points="outliers",
        color_discrete_map={"A1": "#74b9ff", "A2": "#fd79a8", "B": "#55efc4"},
        title="📊 Burnout Score Distribution by Course",
        labels={"burnout_score": "Burnout Score", "course": "Course"},
    )
    box_fig.add_hline(y=65, line_dash="dash", line_color="#e74c3c",
                      annotation_text="High risk threshold",
                      annotation_position="top right")
    box_fig.add_hline(y=40, line_dash="dash", line_color="#f39c12",
                      annotation_text="Moderate threshold",
                      annotation_position="top right")
    box_fig.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font_color="#e0e0e0",
        showlegend=False,
        height=450,
    )

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    def fig_html(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # Build individual user accordion sections
    indiv_html = ""
    for uid, course, fig in individual_figs:
        risk_row = summary_df[summary_df["user_id"] == uid]
        avg  = risk_row["avg_score"].values[0] if not risk_row.empty else 0
        risk_label = "High" if avg >= 65 else "Moderate" if avg >= 40 else "Low"
        badge_color = RISK_COLORS[risk_label]
        indiv_html += f"""
        <details class="user-card">
          <summary>
            <span class="uid">User {uid}</span>
            <span class="course-tag">Course {course}</span>
            <span class="risk-badge" style="background:{badge_color}">
              {risk_label} · avg {avg}
            </span>
          </summary>
          <div class="chart-wrap">{fig_html(fig)}</div>
        </details>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SSAQS Burnout Tracker</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: "Inter", sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      min-height: 100vh;
      color: #e0e0e0;
    }}

    .hero {{
      text-align: center;
      padding: 60px 20px 40px;
      background: linear-gradient(180deg, rgba(224,86,253,0.12) 0%, transparent 100%);
    }}
    .hero h1 {{
      font-size: 2.8rem;
      font-weight: 700;
      background: linear-gradient(90deg, #e056fd, #74b9ff, #55efc4);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 10px;
    }}
    .hero p {{
      color: #b0b0d0;
      font-size: 1.05rem;
      max-width: 650px;
      margin: 0 auto;
      line-height: 1.7;
    }}

    .stats-bar {{
      display: flex;
      justify-content: center;
      gap: 30px;
      flex-wrap: wrap;
      padding: 30px 20px;
    }}
    .stat-card {{
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      padding: 20px 30px;
      text-align: center;
      backdrop-filter: blur(10px);
      min-width: 150px;
    }}
    .stat-card .num {{
      font-size: 2rem;
      font-weight: 700;
      color: #e056fd;
    }}
    .stat-card .label {{
      font-size: 0.85rem;
      color: #a0a0c0;
      margin-top: 4px;
    }}

    .legend-bar {{
      display: flex;
      justify-content: center;
      gap: 20px;
      padding: 0 20px 30px;
      flex-wrap: wrap;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.9rem;
      color: #c0c0d8;
    }}
    .legend-dot {{
      width: 14px; height: 14px;
      border-radius: 50%;
      flex-shrink: 0;
    }}

    .section {{
      max-width: 1300px;
      margin: 0 auto 50px;
      padding: 0 20px;
    }}
    .section-title {{
      font-size: 1.4rem;
      font-weight: 600;
      color: #e0e0f0;
      margin-bottom: 16px;
      padding-bottom: 10px;
      border-bottom: 2px solid rgba(224,86,253,0.3);
    }}

    .chart-wrap {{
      border-radius: 16px;
      overflow: hidden;
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.07);
    }}

    /* Individual user cards */
    .user-grid {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .user-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      overflow: hidden;
      transition: border-color 0.2s;
    }}
    .user-card:hover {{ border-color: rgba(224,86,253,0.4); }}
    .user-card summary {{
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 16px 20px;
      cursor: pointer;
      list-style: none;
      user-select: none;
    }}
    .user-card summary::-webkit-details-marker {{ display: none; }}
    .user-card summary::before {{
      content: "▶";
      font-size: 0.8rem;
      color: #a0a0c0;
      transition: transform 0.25s;
    }}
    .user-card[open] summary::before {{ transform: rotate(90deg); }}

    .uid {{
      font-weight: 600;
      font-size: 1rem;
      color: #e0e0f0;
      min-width: 60px;
    }}
    .course-tag {{
      background: rgba(116,185,255,0.15);
      color: #74b9ff;
      border-radius: 8px;
      padding: 3px 10px;
      font-size: 0.82rem;
      font-weight: 600;
    }}
    .risk-badge {{
      margin-left: auto;
      border-radius: 20px;
      padding: 4px 14px;
      font-size: 0.82rem;
      font-weight: 700;
      color: #fff;
      letter-spacing: 0.3px;
    }}

    .user-card .chart-wrap {{
      border-radius: 0;
      border: none;
      border-top: 1px solid rgba(255,255,255,0.05);
    }}

    footer {{
      text-align: center;
      padding: 30px;
      color: #606080;
      font-size: 0.82rem;
    }}
  </style>
</head>
<body>

<div class="hero">
  <h1>🔥 Burnout Risk Tracker</h1>
  <p>
    University student burnout analysis using the SSAQS dataset —
    combining daily questionnaires, heart rate variability, physical activity, and sleep quality
    into a single evidence-based risk score.
  </p>
</div>

<div class="stats-bar">
  <div class="stat-card">
    <div class="num">{n_users}</div>
    <div class="label">Students tracked</div>
  </div>
  <div class="stat-card">
    <div class="num">{len(all_dates)}</div>
    <div class="label">Days of data</div>
  </div>
  <div class="stat-card">
    <div class="num">{int(summary_df['high_risk_days'].sum())}</div>
    <div class="label">High-risk student-days</div>
  </div>
  <div class="stat-card">
    <div class="num">{summary_df['avg_score'].max():.0f}</div>
    <div class="label">Highest avg score</div>
  </div>
</div>

<div class="legend-bar">
  <div class="legend-item">
    <div class="legend-dot" style="background:#2ecc71"></div>
    Low risk (0–40)
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#f39c12"></div>
    Moderate risk (40–65)
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#e74c3c"></div>
    High risk (65–100)
  </div>
</div>

<div class="section">
  <div class="section-title">🌡️ Cohort Overview — All Students</div>
  <div class="chart-wrap">{fig_html(heat_fig)}</div>
</div>

<div class="section">
  <div class="section-title">📊 Burnout by Course Group</div>
  <div class="chart-wrap">{fig_html(box_fig)}</div>
</div>

<div class="section">
  <div class="section-title">📋 Student Summary</div>
  <div class="chart-wrap">{fig_html(table_fig)}</div>
</div>

<div class="section">
  <div class="section-title">👤 Individual Student Timelines</div>
  <div class="user-grid">
    {indiv_html}
  </div>
</div>

<footer>
  SSAQS Burnout Tracker · Generated from <em>A Dataset of University Students' Stress and Anxiety Levels
  based on Questionnaires and Wearable Sensors</em> · Garcia-Ceja et al. (2026)
</footer>
</body>
</html>"""

    out.write_text(html, encoding="utf-8")
    print(f"\nReport written -> {out.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SSAQS Burnout Tracker")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent / "SSAQS dataset"),
        help="Path to the 'SSAQS dataset' folder",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "burnout_report.html"),
        help="Output HTML file path",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Load course metadata
    users_courses_path = data_dir / "users-courses.csv"
    if not users_courses_path.exists():
        print(f"ERROR: users-courses.csv not found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    users_courses = pd.read_csv(users_courses_path)

    # Discover participant folders (numbered directories)
    user_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    if not user_dirs:
        print(f"ERROR: No numbered participant directories found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(user_dirs)} participant folders in {data_dir}\n")

    all_data: dict[int, pd.DataFrame] = {}
    for user_dir in user_dirs:
        uid = int(user_dir.name)
        print(f"  Processing user {uid:>2} ...", end=" ", flush=True)
        result = compute_burnout(uid, user_dir)
        if result is not None and not result.empty:
            all_data[uid] = result
            avg = result["burnout_score"].mean()
            peak = result["burnout_score"].max()
            print(f"avg={avg:.1f}  peak={peak:.1f}")
        else:
            print("skipped")

    if not all_data:
        print("ERROR: No usable data found for any participant.", file=sys.stderr)
        sys.exit(1)

    print(f"\nBuilding report for {len(all_data)} participants …")
    build_report(all_data, users_courses, out_path)


if __name__ == "__main__":
    main()
