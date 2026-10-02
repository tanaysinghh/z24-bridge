import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (
    DATASET_FACTS,
    SHORT_NAMES,
    SPLITS,
    chance_level,
    model_summary,
    scenario_names,
    split_counts,
)

st.set_page_config(page_title="VibraDiagnose", page_icon="🌉", layout="wide", initial_sidebar_state="collapsed")

INK = "#1f2933"
INK_2 = "#3e4c59"
MUTED = "#6b7683"
FAINT = "#8b97a5"
LINE = "#e3e8ee"
SURFACE = "#ffffff"
SURFACE_2 = "#f4f6f8"
ACCENT = "#2f6f9f"
ACCENT_LIGHT = "#6aa6dc"
WARN = "#c25b3c"
GOOD = "#3f8f6f"
SPLIT_COLORS = {"train": ACCENT, "val": ACCENT_LIGHT, "test": WARN}
METRIC_COLORS = {"test_segment": ACCENT, "test_window": ACCENT_LIGHT}
HEAT_SCALE = [[0.0, "#f4f6f8"], [0.25, "#cfe0ee"], [0.5, "#8db6d8"], [0.75, "#4a86b8"], [1.0, "#1d4f78"]]

TEAM_FILE = Path(__file__).resolve().parent / "team.json"

CSS = f"""
<style>
.stApp {{ background: {SURFACE}; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 5rem; max-width: 1180px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ font-size: 0.98rem; font-weight: 600; padding: 0.6rem 0; margin-right: 1.5rem; color: {MUTED}; background: transparent; }}
.stTabs [data-baseweb="tab"]:hover {{ color: {INK}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT}; height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
.vd-brand {{ font-size: 0.85rem; font-weight: 600; color: {MUTED}; letter-spacing: 0.02em; margin-bottom: 0.6rem; }}
.vd-brand b {{ color: {INK}; }}
.vd-hero {{ font-size: clamp(2.8rem, 6vw, 4.4rem); font-weight: 700; letter-spacing: -0.035em; line-height: 1.02; color: {INK}; margin: 2.2rem 0 0.9rem; }}
.vd-lede {{ font-size: clamp(1.1rem, 1.8vw, 1.35rem); color: {INK_2}; max-width: 46rem; line-height: 1.5; margin: 0 0 2.8rem; }}
.vd-h2 {{ font-size: clamp(1.9rem, 3.4vw, 2.6rem); font-weight: 700; letter-spacing: -0.025em; color: {INK}; margin: 2.2rem 0 0.5rem; line-height: 1.1; }}
.vd-h3 {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 3rem 0 1rem; padding-top: 1.2rem; border-top: 1px solid {LINE}; }}
.vd-sub {{ font-size: 1.08rem; color: {MUTED}; margin: 0 0 2.2rem; max-width: 46rem; line-height: 1.5; }}
.vd-caption {{ font-size: 0.88rem; color: {MUTED}; margin-top: 0.6rem; line-height: 1.5; }}
.vd-grid {{ display: grid; gap: 1rem; }}
.vd-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.vd-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 2rem; }}
.vd-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
@media (max-width: 900px) {{ .vd-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .vd-3, .vd-2 {{ grid-template-columns: 1fr; }} }}
.vd-stat {{ border: 1px solid {LINE}; border-radius: 8px; padding: 1.3rem 1.4rem 1.2rem; background: {SURFACE}; }}
.vd-stat .k {{ font-size: 0.86rem; color: {MUTED}; font-weight: 500; }}
.vd-stat .v {{ font-size: clamp(2.3rem, 4vw, 3.1rem); font-weight: 700; letter-spacing: -0.03em; line-height: 1.05; color: {INK}; margin-top: 0.5rem; font-variant-numeric: tabular-nums; }}
.vd-stat .v.sm {{ font-size: clamp(1.8rem, 2.8vw, 2.3rem); }}
.vd-stat .n {{ font-size: 0.84rem; color: {FAINT}; margin-top: 0.45rem; }}
.vd-stat.hl .v {{ color: {ACCENT}; }}
.vd-col {{ border-top: 1px solid {LINE}; padding-top: 1.1rem; }}
.vd-col .k {{ font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: {MUTED}; }}
.vd-col .t {{ font-size: 1.25rem; font-weight: 700; color: {INK}; margin-top: 0.5rem; line-height: 1.3; }}
.vd-col .d {{ font-size: 1rem; color: {MUTED}; margin-top: 0.35rem; line-height: 1.5; }}
.vd-steps {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0; margin: 0.4rem 0 0.5rem; }}
@media (max-width: 900px) {{ .vd-steps {{ grid-template-columns: 1fr 1fr; row-gap: 1.6rem; }} }}
.vd-step {{ border-top: 2px solid {ACCENT}; padding: 1.1rem 1.6rem 0 0; }}
.vd-step .num {{ font-size: 0.85rem; font-weight: 700; color: {ACCENT}; font-variant-numeric: tabular-nums; }}
.vd-step .t {{ font-size: 1.4rem; font-weight: 700; color: {INK}; margin-top: 0.35rem; letter-spacing: -0.01em; }}
.vd-step .d {{ font-size: 0.98rem; color: {MUTED}; margin-top: 0.4rem; line-height: 1.5; }}
.vd-kv {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; }}
@media (max-width: 900px) {{ .vd-kv {{ grid-template-columns: 1fr 1fr; }} }}
.vd-kv > div {{ padding: 1.1rem 1.2rem; border-right: 1px solid {LINE}; }}
.vd-kv > div:last-child {{ border-right: none; }}
.vd-kv .k {{ font-size: 0.84rem; color: {MUTED}; font-weight: 500; }}
.vd-kv .v {{ font-size: 1.12rem; color: {INK}; font-weight: 700; margin-top: 0.35rem; line-height: 1.35; }}
.vd-best {{ display: grid; grid-template-columns: 1.1fr 1.4fr 1fr 1fr; gap: 1.5rem; align-items: end; border: 1px solid {LINE}; border-left: 3px solid {ACCENT}; border-radius: 8px; padding: 1.7rem 1.9rem; margin-bottom: 2.4rem; }}
@media (max-width: 900px) {{ .vd-best {{ grid-template-columns: 1fr 1fr; }} }}
.vd-best .k {{ font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: {MUTED}; }}
.vd-best .name {{ font-size: 1.9rem; font-weight: 700; color: {INK}; margin-top: 0.3rem; letter-spacing: -0.02em; }}
.vd-best .big {{ font-size: clamp(3rem, 6vw, 4.4rem); font-weight: 700; letter-spacing: -0.04em; line-height: 1; color: {ACCENT}; font-variant-numeric: tabular-nums; }}
.vd-best .mid {{ font-size: clamp(2rem, 3.6vw, 2.6rem); font-weight: 700; letter-spacing: -0.03em; line-height: 1; color: {INK}; font-variant-numeric: tabular-nums; }}
.vd-best .l {{ font-size: 0.9rem; color: {MUTED}; margin-top: 0.5rem; }}
table.vd-table {{ width: 100%; border-collapse: collapse; font-size: 1.05rem; margin: 0.2rem 0 0.4rem; }}
table.vd-table th, table.vd-table td {{ border: none !important; border-bottom: 1px solid {LINE} !important; padding: 0.95rem 0.9rem; text-align: right; font-variant-numeric: tabular-nums; }}
table.vd-table th {{ font-size: 0.82rem; font-weight: 600; color: {MUTED}; background: transparent; }}
table.vd-table th:first-child, table.vd-table td:first-child {{ text-align: left; padding-left: 0; }}
table.vd-table td {{ color: {INK_2}; }}
table.vd-table tr.top td {{ color: {INK}; font-weight: 700; }}
table.vd-table .pm {{ color: {FAINT}; font-size: 0.82rem; font-weight: 400; margin-left: 0.25rem; }}
table.vd-table .tag {{ font-size: 0.72rem; font-weight: 600; color: {ACCENT}; border: 1px solid #c9dbea; border-radius: 4px; padding: 0.1rem 0.4rem; margin-left: 0.6rem; vertical-align: middle; letter-spacing: 0.02em; }}
.vd-list {{ border-top: 2px solid var(--c); padding-top: 1.1rem; }}
.vd-list .t {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin-bottom: 0.7rem; }}
.vd-list ul {{ margin: 0; padding-left: 1.1rem; }}
.vd-list li {{ font-size: 1rem; color: {MUTED}; margin: 0.5rem 0; line-height: 1.5; }}
.vd-list li b {{ color: {INK}; font-weight: 600; }}
.vd-card {{ border: 1px solid {LINE}; border-radius: 8px; padding: 1.5rem 1.6rem; background: {SURFACE}; height: 100%; }}
.vd-card .num {{ font-size: 0.85rem; font-weight: 700; color: {ACCENT}; font-variant-numeric: tabular-nums; }}
.vd-card .t {{ font-size: 1.3rem; font-weight: 700; color: {INK}; margin-top: 0.5rem; letter-spacing: -0.01em; }}
.vd-card .d {{ font-size: 1rem; color: {MUTED}; margin-top: 0.5rem; line-height: 1.55; }}
.vd-pill {{ display: inline-block; font-size: 0.8rem; font-weight: 600; color: {INK_2}; background: {SURFACE_2}; border: 1px solid {LINE}; border-radius: 999px; padding: 0.22rem 0.7rem; margin: 0.9rem 0.35rem 0 0; }}
.vd-road {{ position: relative; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1.2rem; margin: 3.4rem 0 1rem; }}
.vd-road::before {{ content: ""; position: absolute; top: 9px; left: 0; right: 0; height: 2px; background: {LINE}; }}
.vd-road::after {{ content: ""; position: absolute; top: 9px; left: 0; width: 20%; height: 2px; background: {ACCENT}; }}
.vd-ph {{ position: relative; z-index: 1; padding-right: 0.6rem; }}
.vd-ph .dot {{ width: 20px; height: 20px; border-radius: 50%; background: {SURFACE}; border: 2px solid {LINE}; }}
.vd-ph.done .dot {{ background: {ACCENT}; border-color: {ACCENT}; }}
.vd-ph.next .dot {{ border-color: {ACCENT}; box-shadow: 0 0 0 5px rgba(47, 111, 159, 0.14); }}
.vd-ph .s {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {FAINT}; margin-top: 1.2rem; }}
.vd-ph.done .s, .vd-ph.next .s {{ color: {ACCENT}; }}
.vd-ph .t {{ font-size: 1.2rem; font-weight: 700; color: {INK}; margin-top: 0.4rem; line-height: 1.3; }}
.vd-ph.up .t {{ color: {INK_2}; }}
.vd-ph .d {{ font-size: 0.95rem; color: {MUTED}; margin-top: 0.4rem; line-height: 1.5; }}
.vd-here {{ position: absolute; top: -2.2rem; left: 0; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: {ACCENT}; white-space: nowrap; }}
@media (max-width: 900px) {{ .vd-road {{ grid-template-columns: 1fr; gap: 1.8rem; }} .vd-road::before, .vd-road::after {{ display: none; }} .vd-here {{ position: static; display: block; margin-bottom: 0.5rem; }} }}
.vd-team {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1rem; }}
.vd-member {{ border: 1px solid {LINE}; border-radius: 8px; padding: 1.5rem 1.5rem 1.3rem; background: {SURFACE}; }}
.vd-avatar {{ width: 52px; height: 52px; border-radius: 50%; background: #eaf1f7; color: {ACCENT}; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.05rem; letter-spacing: 0.02em; }}
.vd-member .name {{ font-size: 1.25rem; font-weight: 700; color: {INK}; margin-top: 1rem; letter-spacing: -0.01em; }}
.vd-member .role {{ font-size: 0.86rem; color: {MUTED}; margin-top: 0.2rem; }}
.vd-member .tags {{ margin-top: 0.3rem; }}
</style>
"""


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def heading(title, sub=None):
    html(f'<div class="vd-h2">{title}</div>')
    if sub:
        html(f'<div class="vd-sub">{sub}</div>')


def subheading(title):
    html(f'<div class="vd-h3">{title}</div>')


def stat(value, label, note=None, small=False, highlight=False):
    note_html = f'<div class="n">{note}</div>' if note else ""
    cls = "vd-stat hl" if highlight else "vd-stat"
    vcls = "v sm" if small else "v"
    return f'<div class="{cls}"><div class="k">{label}</div><div class="{vcls}">{value}</div>{note_html}</div>'


def base_layout(fig, height, **kwargs):
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="'Source Sans', 'Source Sans Pro', sans-serif", color=INK_2, size=14),
        margin=dict(l=8, r=16, t=36, b=8),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=INK, size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=13, color=INK_2)),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=INK_2))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=INK_2))
    return fig


def show(fig):
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")


def fmt_params(m):
    if m["key"] == "minirocket":
        return "10k kernels"
    return f"{m['params'] / 1000:.0f}k"


def initials(name):
    parts = name.split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()


def render_overview(models, chance):
    best = max(models, key=lambda m: m["test_segment"][0])
    html('<div class="vd-hero">VibraDiagnose</div>')
    html('<div class="vd-lede">Identifying structural damage on the Z24 Bridge from a single accelerometer, using deep learning on real vibration data.</div>')
    html(
        '<div class="vd-grid vd-4">'
        + stat("17", "Damage scenarios", "Progressive damage tests")
        + stat("1,530", "Bridge recordings", "60 s each, real data")
        + stat(str(len(models)), "Models trained", "Three seeds each")
        + stat(f"{best['test_segment'][0]:.1f}%", f"Best accuracy · {best['name']}", f"{best['test_segment'][0] / chance:.1f}× chance on unseen setups", highlight=True)
        + "</div>"
    )
    subheading("At a glance")
    html(
        '<div class="vd-grid vd-3">'
        + '<div class="vd-col"><div class="k">Work completed</div><div class="t">Data, pipeline and three models</div><div class="d">Real Z24 dataset, verified preprocessing, WaveNet, MiniRocket and InceptionTime trained and compared.</div></div>'
        + f'<div class="vd-col"><div class="k">Leading model</div><div class="t">{best["name"]}</div><div class="d">{best["test_segment"][0]:.1f}% per recording, {best["test_window"][0]:.1f}% per 20 s window.</div></div>'
        + '<div class="vd-col"><div class="k">Next up</div><div class="t">Ensemble and hierarchical classification</div><div class="d">Combine the three models and classify damage family before stage.</div></div>'
        + "</div>"
    )


def render_dataset(names):
    heading("Z24 Bridge, Switzerland", "Real vibration data recorded while the bridge was deliberately damaged in 17 controlled steps before demolition.")
    html('<div class="vd-grid vd-4">' + "".join(stat(v, k, n, small=True) for v, k, n in DATASET_FACTS) + "</div>")
    counts = split_counts()
    fig = go.Figure()
    for split, label, setups in SPLITS:
        values = counts[split]
        fig.add_bar(
            y=SHORT_NAMES,
            x=values,
            orientation="h",
            name=f"{label} · {setups.replace('–', '-')}",
            marker=dict(color=SPLIT_COLORS[split], line=dict(color=SURFACE, width=2), cornerradius=3),
            text=[str(values[0])] + [""] * (len(values) - 1),
            textposition="inside",
            insidetextanchor="middle",
            textangle=0,
            constraintext="none",
            textfont=dict(color=INK if split == "val" else "#ffffff", size=12),
            customdata=names,
            hovertemplate="<b>%{customdata}</b><br>" + label + ": %{x} recordings<extra></extra>",
        )
    base_layout(fig, 600, barmode="stack", bargap=0.35)
    fig.update_layout(legend=dict(traceorder="normal"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=13, color=INK), gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)")
    fig.update_xaxes(title=dict(text="Recordings per scenario", font=dict(color=MUTED, size=12)), range=[0, 92], dtick=10)
    subheading("Recordings per damage scenario")
    show(fig)
    html('<div class="vd-caption">Balanced by design: 90 recordings per scenario, split by sensor setup into 60 train, 10 validation and 20 test.</div>')


def render_pipeline(models):
    heading("Pipeline", "From raw vibration to a damage diagnosis.")
    names = [m["name"] for m in models]
    steps = [
        ("01", "Raw data", "1,530 recordings, 27 sensors, 60 s at 100 Hz"),
        ("02", "Preprocessing", "One fixed sensor, band-pass filtered, cut into 20 s windows"),
        ("03", "Model training", f"{', '.join(names[:-1])} and {names[-1]}, three seeds each"),
        ("04", "Evaluation", "Tested only on sensor setups never seen in training"),
    ]
    html(
        '<div class="vd-steps">'
        + "".join(f'<div class="vd-step"><div class="num">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>' for n, t, d in steps)
        + "</div>"
    )
    subheading("Preprocessing")
    kv = [
        ("Channel", "Fixed sensor #22"),
        ("Band-pass", "0.5 – 30 Hz"),
        ("Windows", "20 s, 50% overlap"),
        ("Normalisation", "Z-score per window"),
        ("Split", "Setups 1–6 / 7 / 8–9"),
    ]
    html('<div class="vd-kv">' + "".join(f'<div><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in kv) + "</div>")


def render_results(models, chance):
    best = max(models, key=lambda m: m["test_segment"][0])
    ordered = sorted(models, key=lambda m: -m["test_segment"][0])
    heading("Model comparison", f"Accuracy on held-out sensor setups, mean of three seeds. Chance is {chance:.1f}%.")
    html(
        f'<div class="vd-best"><div><div class="k">Best performer</div><div class="name">{best["name"]}</div></div>'
        f'<div><div class="big">{best["test_segment"][0]:.1f}%</div><div class="l">per 60 s recording</div></div>'
        f'<div><div class="mid">{best["test_window"][0]:.1f}%</div><div class="l">per 20 s window</div></div>'
        f'<div><div class="mid">{best["test_segment"][0] / chance:.1f}×</div><div class="l">better than chance</div></div></div>'
    )
    rows = []
    for m in ordered:
        top = m is best
        tag = '<span class="tag">BEST</span>' if top else ""
        tw, ts, vw = m["test_window"], m["test_segment"], m["val_window"]
        rows.append(
            f'<tr class="{"top" if top else ""}"><td>{m["name"]}{tag}</td><td>{fmt_params(m)}</td>'
            f'<td>{ts[0]:.1f}%<span class="pm">± {ts[1]:.1f}</span></td>'
            f'<td>{tw[0]:.1f}%<span class="pm">± {tw[1]:.1f}</span></td>'
            f'<td>{vw[0]:.1f}%<span class="pm">± {vw[1]:.1f}</span></td></tr>'
        )
    html(
        '<table class="vd-table"><thead><tr><th>Model</th><th>Parameters</th><th>Test, per recording</th><th>Test, per window</th><th>Validation, per window</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )

    fig = go.Figure()
    for key, label in (("test_segment", "Per recording (60 s)"), ("test_window", "Per window (20 s)")):
        fig.add_bar(
            x=[m["name"] for m in ordered],
            y=[m[key][0] for m in ordered],
            error_y=dict(type="data", array=[m[key][1] for m in ordered], color=FAINT, thickness=1.2, width=5),
            name=label,
            marker=dict(color=METRIC_COLORS[key], line=dict(color=SURFACE, width=2), cornerradius=3),
            text=[f"{m[key][0]:.1f}%" for m in ordered],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="#ffffff" if key == "test_segment" else INK, size=15),
            hovertemplate="<b>%{x}</b><br>" + label + ": %{y:.1f}%<extra></extra>",
        )
    fig.add_hline(y=chance, line=dict(color=FAINT, width=1.5, dash="dash"))
    fig.add_annotation(x=1, xref="paper", y=chance, text=f"Chance {chance:.1f}%", showarrow=False, yshift=11, xanchor="right", font=dict(color=MUTED, size=12))
    base_layout(fig, 430, barmode="group", bargap=0.38, bargroupgap=0.06)
    fig.update_yaxes(range=[0, 52], ticksuffix="%", dtick=10)
    fig.update_xaxes(tickfont=dict(size=15, color=INK), gridcolor="rgba(0,0,0,0)")
    subheading("Test accuracy")
    show(fig)

    z = [m["recall"] for m in ordered]
    heat = go.Figure(
        go.Heatmap(
            z=z,
            x=SHORT_NAMES,
            y=[m["name"] for m in ordered],
            colorscale=HEAT_SCALE,
            zmin=0,
            zmax=100,
            xgap=2,
            ygap=2,
            text=[[f"{v:.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(title=dict(text="Recall %", font=dict(color=MUTED, size=12)), tickfont=dict(color=MUTED), thickness=10, outlinewidth=0),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Recall %{z:.0f}%<extra></extra>",
        )
    )
    base_layout(heat, 290)
    heat.update_xaxes(tickangle=-40, tickfont=dict(size=12), gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)")
    heat.update_yaxes(autorange="reversed", tickfont=dict(size=14, color=INK), gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)")
    subheading("Recall per damage scenario, test set")
    show(heat)

    right = [
        "<b>Pier settlement.</b> Deep models reach 75–100% on the 20, 80 and 95 mm stages.",
        "<b>Severe tendon rupture.</b> Six ruptured tendons recognised 90–100% by the deep models.",
        "<b>Damage family.</b> Usually right even when the exact stage is wrong.",
    ]
    wrong = [
        "<b>Neighbouring stages.</b> 2 vs 4 tendons, 2 vs 4 anchor heads.",
        "<b>Reference 1.</b> Read as “6 tendons ruptured”, most likely a temperature effect.",
        "<b>Settlement system installed.</b> Indistinguishable from the undamaged bridge.",
    ]
    html('<div style="height:1.6rem"></div>')
    c1, c2 = st.columns(2, gap="large")
    with c1:
        html(f'<div class="vd-list" style="--c:{GOOD}"><div class="t">What the models get right</div><ul>' + "".join(f"<li>{x}</li>" for x in right) + "</ul></div>")
    with c2:
        html(f'<div class="vd-list" style="--c:{WARN}"><div class="t">Where they struggle</div><ul>' + "".join(f"<li>{x}</li>" for x in wrong) + "</ul></div>")
    html('<div class="vd-caption" style="margin-top:2rem">Prior work (Elios-Lab, IEEE OJIES 2024) reports 60–65% on 5- or 15-class subsets of the ambient data. That protocol is easier and not directly comparable to this 17-class, unseen-setup evaluation.</div>')


def render_decisions():
    heading("Engineering decisions", "Choices that make the results trustworthy.")
    cards = [
        ("01", "Data-driven sensor choice", "Sensors 0–21 move between setups; 22–26 stay fixed. The model uses fixed sensor #22.", "Fixed location"),
        ("02", "A mistake caught and fixed", "The first metric picked a moving sensor (#1). A similarity test exposed it before final training.", "Self-corrected"),
        ("03", "Split by setup, not at random", "Train on setups 1–6, tune on setup 7, test on setups 8–9.", "6 / 1 / 2 setups"),
        ("04", "Stricter than a random split", "Random splits leak slices of the same recording into the test set. This one tests only unseen recordings.", "No leakage"),
    ]
    html(
        '<div class="vd-grid vd-2">'
        + "".join(f'<div class="vd-card"><div class="num">{n}</div><div class="t">{t}</div><div class="d">{d}</div><span class="vd-pill">{p}</span></div>' for n, t, d, p in cards)
        + "</div>"
    )


def render_roadmap():
    heading("Roadmap", "What has been delivered and what comes next.")
    phases = [
        ("done", "Completed", "Data and core models", "Pipeline, WaveNet, MiniRocket, InceptionTime"),
        ("next", "Up next", "Ensemble and hierarchical classification", "Stack the models; classify damage family, then stage"),
        ("up", "Upcoming", "Lifespan estimation", "Remaining useful life (RUL)"),
        ("up", "Upcoming", "Web application", "Interactive diagnosis tool"),
        ("up", "Upcoming", "Integration and testing", "End-to-end system validation"),
    ]
    items = []
    for cls, status, title, desc in phases:
        here = '<div class="vd-here">We are here</div>' if cls == "next" else ""
        items.append(f'<div class="vd-ph {cls}">{here}<div class="dot"></div><div class="s">{status}</div><div class="t">{title}</div><div class="d">{desc}</div></div>')
    html('<div class="vd-road">' + "".join(items) + "</div>")


def render_team(members):
    heading("Team", "Module ownership across the project.")
    cards = []
    for m in members:
        tags = "".join(f'<span class="vd-pill">{t}</span>' for t in m["modules"])
        count = len(m["modules"])
        role = f'{count} module{"s" if count != 1 else ""}'
        cards.append(f'<div class="vd-member"><div class="vd-avatar">{initials(m["name"])}</div><div class="name">{m["name"]}</div><div class="role">{role}</div><div class="tags">{tags}</div></div>')
    html('<div class="vd-team">' + "".join(cards) + "</div>")


def main():
    html(CSS)
    models = model_summary()
    names = scenario_names()
    chance = chance_level()
    members = json.loads(TEAM_FILE.read_text(encoding="utf-8")) if TEAM_FILE.exists() else []
    html('<div class="vd-brand"><b>VibraDiagnose</b> &nbsp;·&nbsp; Z24 Bridge structural health monitoring</div>')
    labels = ["Overview", "Dataset", "Pipeline", "Results", "Decisions", "Roadmap"]
    if members:
        labels.append("👥 Team")
    tabs = st.tabs(labels)
    with tabs[0]:
        render_overview(models, chance)
    with tabs[1]:
        render_dataset(names)
    with tabs[2]:
        render_pipeline(models)
    with tabs[3]:
        render_results(models, chance)
    with tabs[4]:
        render_decisions()
    with tabs[5]:
        render_roadmap()
    if members:
        with tabs[6]:
            render_team(members)


main()
