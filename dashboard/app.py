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

SURFACE = "#1a1a19"
CARD = "#232321"
BORDER = "#34342f"
TEXT = "#ffffff"
TEXT_2 = "#c3c2b7"
MUTED = "#8b8a83"
MODEL_COLORS = {"WaveNet": "#3987e5", "MiniRocket": "#d95926", "InceptionTime": "#199e70"}
SPLIT_COLORS = {"train": "#9085e9", "val": "#c98500", "test": "#d55181"}
GOOD = "#199e70"
BAD = "#e66767"
BLUE_RAMP = [
    [0.0, "#1f2a38"],
    [0.25, "#184f95"],
    [0.5, "#256abf"],
    [0.75, "#5598e7"],
    [1.0, "#b7d3f6"],
]

TEAM_FILE = Path(__file__).resolve().parent / "team.json"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {SURFACE}; color: {TEXT}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {BORDER}; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ font-size: 1.05rem; font-weight: 600; padding: 0.7rem 1.1rem; color: {TEXT_2}; }}
.stTabs [aria-selected="true"] {{ color: {TEXT}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {MODEL_COLORS['WaveNet']}; height: 3px; }}
.eyebrow {{ font-size: 0.95rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: {MODEL_COLORS['WaveNet']}; margin: 1.6rem 0 0.4rem; }}
.hero-title {{ font-size: clamp(2.8rem, 6vw, 4.6rem); font-weight: 800; line-height: 1.02; letter-spacing: -0.03em; margin: 0; }}
.hero-sub {{ font-size: clamp(1.1rem, 2vw, 1.45rem); color: {TEXT_2}; margin: 0.9rem 0 2.2rem; max-width: 52rem; }}
.section-title {{ font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; letter-spacing: -0.02em; margin: 0.2rem 0 0.3rem; }}
.section-sub {{ font-size: 1.15rem; color: {TEXT_2}; margin: 0 0 1.6rem; }}
.grid {{ display: grid; gap: 1rem; margin-bottom: 1.4rem; }}
.g4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.g3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.g2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.g5 {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
@media (max-width: 900px) {{ .g4, .g5 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .g3, .g2 {{ grid-template-columns: 1fr; }} }}
.card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 1.5rem 1.6rem; }}
.stat-num {{ font-size: clamp(2.4rem, 4.5vw, 3.6rem); font-weight: 800; line-height: 1; letter-spacing: -0.03em; color: {TEXT}; }}
.stat-num.sm {{ font-size: clamp(1.9rem, 3vw, 2.6rem); }}
.stat-label {{ font-size: 1rem; color: {TEXT_2}; margin-top: 0.6rem; font-weight: 500; }}
.stat-note {{ font-size: 0.85rem; color: {MUTED}; margin-top: 0.35rem; }}
.accent {{ border-top: 4px solid var(--c, {MODEL_COLORS['WaveNet']}); }}
.best {{ background: linear-gradient(135deg, #1c3354 0%, {CARD} 70%); border: 1px solid #2c5a92; border-radius: 22px; padding: 2rem 2.2rem; display: flex; gap: 2.5rem; align-items: center; flex-wrap: wrap; margin-bottom: 1.6rem; }}
.best .tag {{ font-size: 0.85rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #86b6ef; }}
.best .name {{ font-size: 2.2rem; font-weight: 800; margin-top: 0.2rem; }}
.best .big {{ font-size: clamp(3.4rem, 7vw, 5.2rem); font-weight: 800; letter-spacing: -0.04em; line-height: 1; }}
.best .lbl {{ font-size: 1rem; color: {TEXT_2}; margin-top: 0.4rem; }}
.flow {{ display: flex; align-items: stretch; gap: 0; margin: 0.6rem 0 2rem; flex-wrap: wrap; }}
.flow .step {{ flex: 1 1 180px; background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 1.4rem 1.3rem; border-top: 4px solid var(--c); }}
.flow .icon {{ font-size: 2.2rem; }}
.flow .name {{ font-size: 1.35rem; font-weight: 800; margin: 0.5rem 0 0.4rem; }}
.flow .meta {{ font-size: 0.98rem; color: {TEXT_2}; line-height: 1.5; }}
.flow .arrow {{ display: flex; align-items: center; justify-content: center; font-size: 2rem; color: {MUTED}; padding: 0 0.6rem; }}
@media (max-width: 900px) {{ .flow .arrow {{ width: 100%; padding: 0.3rem 0; transform: rotate(90deg); }} }}
.chip-row {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.9rem; }}
@media (max-width: 900px) {{ .chip-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.chip {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.1rem 1.1rem; text-align: center; }}
.chip .icon {{ font-size: 1.9rem; }}
.chip .k {{ font-size: 1.05rem; font-weight: 700; margin-top: 0.4rem; }}
.chip .v {{ font-size: 0.92rem; color: {TEXT_2}; margin-top: 0.25rem; }}
table.res {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 1.15rem; margin-bottom: 1.6rem; }}
table.res th, table.res td {{ border-left: none !important; border-right: none !important; border-top: none !important; }}
table.res th {{ text-align: right; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; color: {MUTED}; font-weight: 700; padding: 0.8rem 1rem; border-bottom: 1px solid {BORDER}; }}
table.res th:first-child, table.res td:first-child {{ text-align: left; }}
table.res td {{ text-align: right; padding: 1.05rem 1rem; border-bottom: 1px solid {BORDER}; font-variant-numeric: tabular-nums; }}
table.res tr.top td {{ background: rgba(57, 135, 229, 0.10); font-weight: 700; }}
table.res .dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 0.6rem; vertical-align: middle; }}
table.res .pm {{ color: {MUTED}; font-size: 0.85rem; font-weight: 500; }}
.callout {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 5px solid var(--c); border-radius: 16px; padding: 1.4rem 1.6rem; height: 100%; }}
.callout h4 {{ font-size: 1.35rem; font-weight: 800; margin: 0 0 0.8rem; }}
.callout ul {{ margin: 0; padding-left: 1.2rem; }}
.callout li {{ font-size: 1.08rem; color: {TEXT_2}; margin: 0.45rem 0; }}
.callout li b {{ color: {TEXT}; }}
.dcard {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 1.6rem 1.6rem; height: 100%; }}
.dcard .icon {{ font-size: 2.3rem; }}
.dcard .t {{ font-size: 1.4rem; font-weight: 800; margin: 0.7rem 0 0.5rem; }}
.dcard .d {{ font-size: 1.05rem; color: {TEXT_2}; line-height: 1.5; }}
.dcard .pill {{ display: inline-block; margin-top: 0.9rem; padding: 0.3rem 0.75rem; border-radius: 999px; font-size: 0.85rem; font-weight: 700; background: #2a2a27; color: {TEXT_2}; }}
.timeline {{ position: relative; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1rem; margin: 2.4rem 0 1rem; }}
.timeline::before {{ content: ""; position: absolute; top: 27px; left: 10%; right: 10%; height: 4px; background: {BORDER}; border-radius: 2px; }}
.timeline::after {{ content: ""; position: absolute; top: 27px; left: 10%; width: 20%; height: 4px; background: {GOOD}; border-radius: 2px; }}
.tl {{ position: relative; text-align: center; z-index: 1; }}
.tl .node {{ width: 58px; height: 58px; margin: 0 auto; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 800; background: {CARD}; border: 3px solid {BORDER}; color: {MUTED}; }}
.tl.done .node {{ background: {GOOD}; border-color: {GOOD}; color: #0f2a20; }}
.tl.next .node {{ border-color: {MODEL_COLORS['WaveNet']}; color: {TEXT}; box-shadow: 0 0 0 6px rgba(57, 135, 229, 0.18); }}
.tl .status {{ margin-top: 1rem; font-size: 0.8rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: {MUTED}; }}
.tl.done .status {{ color: {GOOD}; }}
.tl.next .status {{ color: #86b6ef; }}
.tl .name {{ margin-top: 0.4rem; font-size: 1.15rem; font-weight: 800; }}
.tl .sub {{ margin-top: 0.35rem; font-size: 0.92rem; color: {TEXT_2}; }}
.here {{ position: absolute; top: -2.1rem; left: 50%; transform: translateX(-50%); font-size: 0.78rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; background: {MODEL_COLORS['WaveNet']}; color: #fff; padding: 0.25rem 0.65rem; border-radius: 999px; white-space: nowrap; }}
@media (max-width: 900px) {{ .timeline {{ grid-template-columns: 1fr; gap: 1.8rem; }} .timeline::before, .timeline::after {{ display: none; }} .here {{ position: static; transform: none; display: inline-block; margin-bottom: 0.6rem; }} }}
.team {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 1rem; }}
.member {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 1.6rem; text-align: center; }}
.avatar {{ width: 72px; height: 72px; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; font-weight: 800; background: #2a3a52; color: #b7d3f6; }}
.member .n {{ font-size: 1.25rem; font-weight: 800; margin-top: 0.9rem; }}
.member .r {{ font-size: 0.98rem; color: {TEXT_2}; margin-top: 0.35rem; line-height: 1.45; }}
.note {{ font-size: 0.9rem; color: {MUTED}; margin-top: 0.4rem; }}
</style>
"""


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def section(eyebrow, title, sub=None):
    html(f'<div class="eyebrow">{eyebrow}</div><div class="section-title">{title}</div>')
    if sub:
        html(f'<div class="section-sub">{sub}</div>')


def stat_card(value, label, note=None, color=None, small=False):
    style = f' style="--c:{color}"' if color else ""
    cls = "card accent" if color else "card"
    note_html = f'<div class="stat-note">{note}</div>' if note else ""
    num_cls = "stat-num sm" if small else "stat-num"
    return f'<div class="{cls}"{style}><div class="{num_cls}">{value}</div><div class="stat-label">{label}</div>{note_html}</div>'


def base_layout(fig, height, **kwargs):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_2, size=15),
        margin=dict(l=10, r=20, t=40, b=10),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT, size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=14, color=TEXT_2)),
        **kwargs,
    )
    fig.update_xaxes(gridcolor="#2c2c29", zerolinecolor="#3a3a36", linecolor=BORDER, tickfont=dict(color=TEXT_2))
    fig.update_yaxes(gridcolor="#2c2c29", zerolinecolor="#3a3a36", linecolor=BORDER, tickfont=dict(color=TEXT_2))
    return fig


def show(fig):
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")


def fmt_params(m):
    if m["key"] == "minirocket":
        return "10k kernels"
    return f"{m['params'] / 1000:.0f}k"


def render_overview(models, chance):
    best = max(models, key=lambda m: m["test_segment"][0])
    html('<div class="eyebrow">Structural health monitoring · Z24 Bridge</div>')
    html('<div class="hero-title">VibraDiagnose</div>')
    html('<div class="hero-sub">Identifying bridge damage from a single accelerometer with deep learning.</div>')
    html(
        '<div class="grid g4">'
        + stat_card("17", "Damage scenarios", color="#9085e9")
        + stat_card("1,530", "Real bridge recordings", color="#c98500")
        + stat_card(str(len(models)), "Models trained", note="× 3 seeds each", color="#d55181")
        + stat_card(
            f"{best['test_segment'][0]:.1f}%",
            f"Best accuracy · {best['name']}",
            note=f"{best['test_segment'][0] / chance:.1f}× chance · unseen setups",
            color=MODEL_COLORS[best["name"]],
        )
        + "</div>"
    )
    html('<div class="eyebrow">At a glance</div>')
    html(
        '<div class="grid g3">'
        + '<div class="dcard"><div class="icon">✅</div><div class="t">Work completed</div><div class="d">Real dataset · pipeline · 3 models trained & compared</div></div>'
        + '<div class="dcard"><div class="icon">🏆</div><div class="t">Leading model</div><div class="d">'
        + f"{best['name']} · {best['test_window'][0]:.1f}% per 20 s window · {best['test_segment'][0]:.1f}% per recording</div></div>"
        + '<div class="dcard"><div class="icon">🧭</div><div class="t">Next up</div><div class="d">Ensemble & hierarchical classification</div></div>'
        + "</div>"
    )


def render_dataset(names):
    section("The data", "Z24 Bridge, Switzerland", "Real vibration data recorded while the bridge was deliberately damaged in 17 controlled steps.")
    html('<div class="grid g4">' + "".join(stat_card(v, k, small=True) for v, k in DATASET_FACTS) + "</div>")
    counts = split_counts()
    fig = go.Figure()
    for split, label, setups in SPLITS:
        fig.add_bar(
            y=SHORT_NAMES,
            x=counts[split],
            orientation="h",
            name=f"{label} ({setups})",
            marker=dict(color=SPLIT_COLORS[split], line=dict(color=SURFACE, width=2), cornerradius=4),
            customdata=names,
            hovertemplate="<b>%{customdata}</b><br>" + label + ": %{x} recordings<extra></extra>",
        )
    base_layout(fig, 640, barmode="stack", bargap=0.28)
    fig.update_layout(legend=dict(traceorder="normal"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=14, color=TEXT), gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title=dict(text="Recordings per scenario", font=dict(color=MUTED, size=13)), range=[0, 95], dtick=10)
    html('<div class="eyebrow">Recordings per damage scenario</div>')
    show(fig)
    html('<div class="note">Perfectly balanced: 90 recordings per scenario (60 train · 10 validation · 20 test).</div>')


def render_pipeline(models):
    section("How it works", "From raw vibration to diagnosis")
    steps = [
        ("📡", "Raw data", "1,530 recordings<br>27 sensors × 60 s", "#9085e9"),
        ("🧹", "Preprocessing", "1 sensor · filtered<br>20 s windows", "#c98500"),
        ("🧠", "Model training", " · ".join(m["name"] for m in models) + "<br>3 seeds each", "#d55181"),
        ("📊", "Results", "Tested on setups<br>never seen in training", GOOD),
    ]
    parts = []
    for i, (icon, name, meta, color) in enumerate(steps):
        if i:
            parts.append('<div class="arrow">→</div>')
        parts.append(f'<div class="step" style="--c:{color}"><div class="icon">{icon}</div><div class="name">{name}</div><div class="meta">{meta}</div></div>')
    html('<div class="flow">' + "".join(parts) + "</div>")
    html('<div class="eyebrow">Preprocessing steps</div>')
    chips = [
        ("🎯", "Channel selection", "Fixed sensor #22"),
        ("🎚️", "Band-pass", "0.5 – 30 Hz"),
        ("✂️", "Windowing", "20 s · 50% overlap"),
        ("📏", "Normalisation", "Z-score per window"),
        ("🧪", "Split by setup", "Train 1–6 · Val 7 · Test 8–9"),
    ]
    html('<div class="chip-row">' + "".join(f'<div class="chip"><div class="icon">{i}</div><div class="k">{k}</div><div class="v">{v}</div></div>' for i, k, v in chips) + "</div>")


def render_results(models, chance):
    best = max(models, key=lambda m: m["test_segment"][0])
    section("Results", "Model comparison", "Accuracy on held-out bridge setups · mean of 3 seeds · chance = 5.9%")
    html(
        f'<div class="best"><div><div class="tag">🏆 Best performer</div><div class="name">{best["name"]}</div></div>'
        f'<div><div class="big">{best["test_segment"][0]:.1f}%</div><div class="lbl">per 60 s recording</div></div>'
        f'<div><div class="big" style="font-size:clamp(2.4rem,4.5vw,3.4rem)">{best["test_window"][0]:.1f}%</div><div class="lbl">per 20 s window</div></div>'
        f'<div><div class="big" style="font-size:clamp(2.4rem,4.5vw,3.4rem)">{best["test_segment"][0] / chance:.1f}×</div><div class="lbl">better than chance</div></div></div>'
    )
    ordered = sorted(models, key=lambda m: -m["test_segment"][0])
    rows = []
    for m in ordered:
        cls = ' class="top"' if m is best else ""
        tw, ts, vw = m["test_window"], m["test_segment"], m["val_window"]
        rows.append(
            f'<tr{cls}><td><span class="dot" style="background:{MODEL_COLORS[m["name"]]}"></span>{m["name"]}</td>'
            f"<td>{fmt_params(m)}</td>"
            f'<td>{ts[0]:.1f}% <span class="pm">± {ts[1]:.1f}</span></td>'
            f'<td>{tw[0]:.1f}% <span class="pm">± {tw[1]:.1f}</span></td>'
            f'<td>{vw[0]:.1f}% <span class="pm">± {vw[1]:.1f}</span></td></tr>'
        )
    html(
        '<table class="res"><thead><tr><th>Model</th><th>Params</th><th>Test · recording</th><th>Test · window</th><th>Validation · window</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )

    fig = go.Figure()
    metrics = [("test_segment", "Per recording (60 s)"), ("test_window", "Per window (20 s)")]
    for m in ordered:
        fig.add_bar(
            x=[label for _, label in metrics],
            y=[m[k][0] for k, _ in metrics],
            error_y=dict(type="data", array=[m[k][1] for k, _ in metrics], color=TEXT_2, thickness=1.5, width=6),
            name=m["name"],
            marker=dict(color=MODEL_COLORS[m["name"]], line=dict(color=SURFACE, width=2), cornerradius=4),
            text=[f"{m[k][0]:.1f}%" for k, _ in metrics],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="#ffffff", size=17),
            hovertemplate=f"<b>{m['name']}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        )
    fig.add_hline(y=chance, line=dict(color=MUTED, width=2, dash="dash"))
    fig.add_annotation(x=1.5, y=chance, text=f"chance {chance:.1f}%", showarrow=False, yshift=12, font=dict(color=MUTED, size=13), xref="x", yref="y")
    base_layout(fig, 470, barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 55], ticksuffix="%", dtick=10)
    fig.update_xaxes(tickfont=dict(size=16, color=TEXT), gridcolor="rgba(0,0,0,0)")
    html('<div class="eyebrow">Test accuracy</div>')
    show(fig)

    z = [m["recall"] for m in ordered]
    heat = go.Figure(
        go.Heatmap(
            z=z,
            x=SHORT_NAMES,
            y=[m["name"] for m in ordered],
            colorscale=BLUE_RAMP,
            zmin=0,
            zmax=100,
            xgap=3,
            ygap=3,
            text=[[f"{v:.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorbar=dict(title=dict(text="Recall %", font=dict(color=TEXT_2)), tickfont=dict(color=TEXT_2), thickness=12),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Recall: %{z:.0f}%<extra></extra>",
        )
    )
    base_layout(heat, 300)
    heat.update_xaxes(tickangle=-40, tickfont=dict(size=13), gridcolor="rgba(0,0,0,0)")
    heat.update_yaxes(autorange="reversed", tickfont=dict(size=15, color=TEXT), gridcolor="rgba(0,0,0,0)")
    html('<div class="eyebrow">Which scenarios each model recognises · test recall</div>')
    show(heat)

    right = [
        "<b>Pier settlement</b> — deep models hit 75–100% on the 20, 80 and 95 mm stages",
        "<b>Severe tendon rupture</b> — deep models catch 6 ruptured tendons 90–100%",
        "<b>Damage family</b> usually right, even when the exact stage is wrong",
    ]
    wrong = [
        "<b>Neighbouring stages</b> — 2 vs 4 tendons, 2 vs 4 anchor heads",
        "<b>Reference 1</b> mistaken for “6 tendons ruptured” — likely temperature effects",
        "<b>Settlement system installed</b> — looks identical to the undamaged bridge",
    ]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(f'<div class="callout" style="--c:{GOOD}"><h4>✅ What the models get right</h4><ul>' + "".join(f"<li>{x}</li>" for x in right) + "</ul></div>")
    with c2:
        html(f'<div class="callout" style="--c:{BAD}"><h4>⚠️ What they get wrong</h4><ul>' + "".join(f"<li>{x}</li>" for x in wrong) + "</ul></div>")
    html('<div class="note" style="margin-top:1.2rem">Prior work (Elios-Lab, IEEE OJIES 2024): 60–65% on 5- or 15-class subsets of the ambient data — a different, easier protocol than our 17-class, unseen-setup test.</div>')


def render_decisions():
    section("Engineering decisions", "Choices that make the results trustworthy")
    cards = [
        ("🎯", "Data-driven sensor choice", "Sensors 0–21 move between setups. 22–26 stay fixed. We use #22.", "Fixed location"),
        ("🔁", "We caught our own mistake", "First metric picked a moving sensor (#1). A similarity test exposed it — fixed before final training.", "Self-corrected"),
        ("🧪", "Split by setup, not at random", "Train on setups 1–6 · tune on 7 · test on 8–9.", "6 / 1 / 2 setups"),
        ("🛡️", "Stricter than a random split", "Random splits leak slices of the same recording into the test set. Ours tests only unseen recordings.", "No leakage"),
    ]
    html(
        '<div class="grid g2">'
        + "".join(f'<div class="dcard"><div class="icon">{i}</div><div class="t">{t}</div><div class="d">{d}</div><div class="pill">{p}</div></div>' for i, t, d, p in cards)
        + "</div>"
    )


def render_roadmap():
    section("Roadmap", "What's next")
    phases = [
        ("done", "✓", "Data & core models", "Pipeline + WaveNet · MiniRocket · InceptionTime"),
        ("next", "2", "Ensemble & hierarchical classification", "Combine models · damage family → stage"),
        ("", "3", "Lifespan (RUL)", "Remaining useful life estimation"),
        ("", "4", "Web application", "Interactive diagnosis tool"),
        ("", "5", "Integration & testing", "End-to-end system validation"),
    ]
    status = {"done": "Completed", "next": "Up next", "": "Upcoming"}
    items = []
    for cls, node, name, sub in phases:
        here = '<div class="here">We are here</div>' if cls == "next" else ""
        items.append(f'<div class="tl {cls}">{here}<div class="node">{node}</div><div class="status">{status[cls]}</div><div class="name">{name}</div><div class="sub">{sub}</div></div>')
    html('<div class="timeline">' + "".join(items) + "</div>")


def render_team():
    section("Team", "Who owns what")
    members = json.loads(TEAM_FILE.read_text(encoding="utf-8")) if TEAM_FILE.exists() else []
    cards = []
    for m in members:
        initials = "".join(p[0] for p in m["name"].split()[:2]).upper()
        cards.append(f'<div class="member"><div class="avatar">{initials}</div><div class="n">{m["name"]}</div><div class="r">{" · ".join(m["modules"])}</div></div>')
    html('<div class="team">' + "".join(cards) + "</div>")


def main():
    html(CSS)
    models = model_summary()
    names = scenario_names()
    chance = chance_level()
    labels = ["🌉 Overview", "📦 Dataset", "🔀 Pipeline", "📊 Results", "🧩 Decisions", "🗺️ Roadmap"]
    if TEAM_FILE.exists():
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
    if TEAM_FILE.exists():
        with tabs[6]:
            render_team()


main()
