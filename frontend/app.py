# frontend/app.py — EvalOps Dashboard (Replit edition)
# All data comes from the Railway backend API — no direct file access.

import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime

# ── Backend URL ───────────────────────────────────────────────────────────────
# Set BACKEND_URL in Replit Secrets:  https://your-app.up.railway.app
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EvalOps · JobRex",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
INDIGO   = "#6366f1"
INDIGO_L = "#818cf8"
EMERALD  = "#10b981"
ROSE     = "#f43f5e"
AMBER    = "#f59e0b"
SLATE_1  = "#0f172a"
SLATE_2  = "#1e293b"
SLATE_3  = "#334155"
SLATE_4  = "#475569"
SLATE_5  = "#94a3b8"
BG_PAGE  = "#f8fafc"
BG_CARD  = "#ffffff"
BORDER   = "#e2e8f0"

st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: {BG_PAGE}; }}
  [data-testid="block-container"] {{ padding: 0 2.5rem 2rem 2.5rem !important; max-width: 1400px; }}
  h1,h2,h3,h4 {{ color:{SLATE_1}; font-family:'Inter','Segoe UI',sans-serif; }}
  p, li, span {{ font-family:'Inter','Segoe UI',sans-serif; color:{SLATE_3}; }}
  .eval-header {{ display:flex; align-items:center; justify-content:space-between;
      padding:1.4rem 0 1rem 0; border-bottom:1px solid {BORDER}; margin-bottom:1.5rem; }}
  .eval-logo {{ display:flex; align-items:center; gap:0.6rem; }}
  .eval-logo-icon {{ width:36px; height:36px; background:{INDIGO}; border-radius:8px;
      display:flex; align-items:center; justify-content:center; font-size:18px; color:white; }}
  .eval-logo-text {{ font-size:1.25rem; font-weight:700; color:{SLATE_1}; }}
  .eval-logo-sub  {{ font-size:0.75rem; color:{SLATE_5}; font-weight:500; letter-spacing:0.05em; text-transform:uppercase; }}
  .eval-badge {{ background:{EMERALD}18; color:{EMERALD}; border:1px solid {EMERALD}40;
      border-radius:999px; padding:3px 12px; font-size:0.72rem; font-weight:600;
      letter-spacing:0.04em; text-transform:uppercase; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:1rem; margin-bottom:1.5rem; }}
  .kpi-card {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px;
      padding:1.1rem 1.2rem; position:relative; overflow:hidden; }}
  .kpi-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
  .kpi-card.indigo::before  {{ background:{INDIGO}; }}
  .kpi-card.emerald::before {{ background:{EMERALD}; }}
  .kpi-card.rose::before    {{ background:{ROSE}; }}
  .kpi-card.amber::before   {{ background:{AMBER}; }}
  .kpi-card.slate::before   {{ background:{SLATE_4}; }}
  .kpi-label {{ font-size:0.72rem; color:{SLATE_5}; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem; }}
  .kpi-value {{ font-size:1.85rem; font-weight:700; color:{SLATE_1}; line-height:1; }}
  .kpi-sub   {{ font-size:0.75rem; color:{SLATE_5}; margin-top:0.35rem; }}
  .section-header {{ font-size:0.8rem; font-weight:700; text-transform:uppercase;
      letter-spacing:0.08em; color:{SLATE_5}; margin:1.6rem 0 0.8rem 0;
      display:flex; align-items:center; gap:0.5rem; }}
  .section-header::after {{ content:''; flex:1; height:1px; background:{BORDER}; }}
  .card {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px; padding:1.25rem 1.4rem; }}
  .card-title   {{ font-size:0.92rem; font-weight:600; color:{SLATE_1}; margin-bottom:0.25rem; }}
  .card-caption {{ font-size:0.75rem; color:{SLATE_5}; margin-bottom:1rem; }}
  .badge        {{ display:inline-block; border-radius:999px; padding:2px 10px; font-size:0.72rem; font-weight:600; }}
  .badge-pass   {{ background:{EMERALD}18; color:{EMERALD}; border:1px solid {EMERALD}40; }}
  .badge-fail   {{ background:{ROSE}15;    color:{ROSE};    border:1px solid {ROSE}40;    }}
  .badge-easy   {{ background:{INDIGO}15;  color:{INDIGO};  border:1px solid {INDIGO}40;  }}
  .badge-hard   {{ background:{AMBER}15;   color:{AMBER};   border:1px solid {AMBER}40;   }}
  .badge-ok     {{ background:{EMERALD}15; color:{EMERALD}; border:1px solid {EMERALD}40; }}
  .inspector-box {{ background:{BG_PAGE}; border:1px solid {BORDER}; border-radius:10px;
      padding:1rem 1.2rem; font-size:0.85rem; color:{SLATE_3}; }}
  .inspector-label {{ font-size:0.7rem; font-weight:600; text-transform:uppercase;
      letter-spacing:0.06em; color:{SLATE_5}; margin-bottom:0.3rem; }}
  [data-testid="stTabs"] button {{ font-family:'Inter',sans-serif !important;
      font-size:0.82rem !important; font-weight:600 !important; color:{SLATE_4} !important; }}
  [data-testid="stTabs"] button[aria-selected="true"] {{ color:{INDIGO} !important; border-bottom-color:{INDIGO} !important; }}
  [data-testid="stButton"] button {{ background:{INDIGO} !important; color:white !important;
      border:none !important; border-radius:8px !important; font-family:'Inter',sans-serif !important; font-weight:600 !important; }}
  [data-testid="stButton"] button:hover {{ background:{INDIGO_L} !important; }}
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    plot_bgcolor=BG_CARD, paper_bgcolor=BG_CARD,
    font=dict(family="Inter, Segoe UI, sans-serif", color=SLATE_3, size=12),
    margin=dict(t=16, b=40, l=12, r=12),
    xaxis=dict(showgrid=False, linecolor=BORDER, tickcolor=BORDER, tickfont=dict(size=11, color=SLATE_5)),
    hoverlabel=dict(bgcolor=SLATE_2, font=dict(color="white", size=12), bordercolor=SLATE_2),
)
BASE_YAXIS = dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(size=11, color=SLATE_5))

# ── API helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    try:
        r = requests.get(f"{BACKEND_URL}/results", timeout=10)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.sidebar.warning(f"Backend: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_policy() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/policy", timeout=10)
        r.raise_for_status()
        return r.json().get("policy", "Policy not found.")
    except Exception as e:
        return f"Could not load policy: {e}"

def append_to_policy_api(rule: str, author: str) -> dict:
    r = requests.post(f"{BACKEND_URL}/policy", json={"rule": rule, "author": author}, timeout=10)
    r.raise_for_status()
    return r.json()

def run_agent_api(prompt: str) -> dict:
    r = requests.post(f"{BACKEND_URL}/agent", json={"prompt": prompt}, timeout=120)
    r.raise_for_status()
    return r.json()

def trigger_evals_api() -> dict:
    r = requests.post(f"{BACKEND_URL}/evals", timeout=10)
    r.raise_for_status()
    return r.json()

def evals_status_api() -> dict:
    r = requests.get(f"{BACKEND_URL}/evals/status", timeout=5)
    r.raise_for_status()
    return r.json()

def fmt_violation(v):
    if not v or v in ("—", "None", "Parse_Error"):
        return v or "—"
    return v.replace("_", " ")

def kpi(label, value, sub="", color="indigo"):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    return f"<div class='kpi-card {color}'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>{sub_html}</div>"

# ── Header ─────────────────────────────────────────────────────────────────────
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=4).json()
    badge_text = "● Live"
    badge_style = ""
except Exception:
    badge_text = "● Backend offline"
    badge_style = f"background:{ROSE}15;color:{ROSE};border-color:{ROSE}40;"

st.markdown(f"""
<div class="eval-header">
  <div class="eval-logo">
    <div class="eval-logo-icon">🛡️</div>
    <div>
      <div class="eval-logo-text">EvalOps</div>
      <div class="eval-logo-sub">by JobRex · Flight Agent</div>
    </div>
  </div>
  <div class="eval-badge" style="{badge_style}">{badge_text}</div>
</div>
""", unsafe_allow_html=True)

# Sidebar — eval controls
with st.sidebar:
    st.markdown("### Run Evals")
    if st.button("▶ Trigger eval run"):
        try:
            resp = trigger_evals_api()
            if resp.get("status") == "already_running":
                st.warning(f"Already running (run_id: {resp.get('run_id')})")
            else:
                st.success(f"Started! run_id: {resp.get('run_id')}")
        except Exception as e:
            st.error(str(e))
    try:
        status = evals_status_api()
        if status.get("running"):
            st.info(f"🔄 Running... (run_id: {status.get('run_id')})")
        elif status.get("last_completed"):
            st.caption(f"Last run: {status['last_completed']}")
        if status.get("last_error"):
            st.error(f"Last error: {status['last_error']}")
    except Exception:
        pass
    st.markdown("---")
    st.caption(f"Backend: `{BACKEND_URL}`")
    if st.button("🔄 Refresh data"):
        load_data.clear()
        load_policy.clear()
        st.rerun()

df = load_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Evaluation", "KA2L Routing", "Playground", "Policy Editor", "Regression",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if df.empty:
        st.info("No evaluation data yet. Use the sidebar to trigger an eval run.")
        st.stop()

    total        = len(df)
    passed       = int((df["binary_consistency"] == 1).sum())
    hallucinated = int((df["hallucination_binary"] == 1).sum())
    violations   = int((df["policy_violation"].notna() & (df["policy_violation"] != "None")).sum())
    avg_faith    = df["faithfulness_score"].replace(-1, float("nan")).mean()
    policy_hash  = df["policy_hash"].iloc[0] if "policy_hash" in df.columns else "—"

    st.markdown(f"""
    <div class="kpi-grid">
      {kpi("Consistency Rate", f"{(passed/total)*100:.0f}%", f"{passed} of {total} passes", "indigo")}
      {kpi("Avg Faithfulness", f"{avg_faith:.2f}" if not pd.isna(avg_faith) else "—", "1.0 = perfect policy alignment", "emerald")}
      {kpi("Hallucinations", str(hallucinated), "cases with invented facts", "rose" if hallucinated else "emerald")}
      {kpi("Policy Violations", str(violations), "rule breaks by judge", "amber" if violations else "emerald")}
      {kpi("Total Cases", str(total), f"policy {policy_hash}", "slate")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Evaluation Results</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card"><div class="card-title">Pass / Fail by Case</div><div class="card-caption">Green = matched expected verdict · Red = wrong</div>', unsafe_allow_html=True)
        cdf = df[["id", "binary_consistency", "input"]].copy()
        cdf["color"] = cdf["binary_consistency"].map({1: EMERALD, 0: ROSE, -1: SLATE_5})
        cdf["label"] = cdf["binary_consistency"].map({1: "PASS", 0: "FAIL", -1: "?"})
        fig = go.Figure()
        for _, r in cdf.iterrows():
            fig.add_trace(go.Bar(x=[f"Case {r['id']}"], y=[1], marker_color=r["color"],
                text=r["label"], textposition="inside", textfont=dict(size=13, color="white"),
                hovertext=str(r["input"])[:60], showlegend=False))
        fig.update_layout(**CHART_LAYOUT, height=240, yaxis={**BASE_YAXIS, "visible": False, "range": [0,1.3]}, bargap=0.35)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">Faithfulness Score</div><div class="card-caption">Policy alignment rated by Groq 70b judge · 0.8 = good</div>', unsafe_allow_html=True)
        fdf = df[["id", "faithfulness_score"]].copy()
        fdf["fs"] = fdf["faithfulness_score"].replace(-1, None)
        fdf["color"] = fdf["fs"].apply(lambda v: EMERALD if v is not None and v >= 0.8 else AMBER if v is not None and v >= 0.5 else ROSE if v is not None else SLATE_5)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=[f"Case {i}" for i in fdf["id"]], y=fdf["fs"].fillna(0),
            marker_color=fdf["color"],
            text=[f"{v:.2f}" if v is not None else "—" for v in fdf["fs"]],
            textposition="outside", textfont=dict(size=12, color=SLATE_3)))
        fig2.add_hline(y=0.8, line_dash="dash", line_color=EMERALD, annotation_text="0.8 good", annotation_font_color=EMERALD, annotation_position="top right")
        fig2.add_hline(y=0.5, line_dash="dot", line_color=AMBER, annotation_text="0.5 warn", annotation_font_color=AMBER, annotation_position="top right")
        fig2.update_layout(**CHART_LAYOUT, height=240, yaxis={**BASE_YAXIS, "range": [0,1.25]}, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Case Detail</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown('<div class="card"><div class="card-title">Health Summary</div><div class="card-caption">Hallucinations, violations, tool status</div>', unsafe_allow_html=True)
        rows = [{"Case": f"Case {r['id']}", "Halluc.": "🔴 Yes" if r.get("hallucination_binary")==1 else "🟢 No",
                 "Violation": fmt_violation(r.get("policy_violation","None")), "Tool": r.get("tool_status","—"), "⏱ s": r.get("duration","—")}
                for _, r in df.iterrows()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card"><div class="card-title">Expected vs Actual</div><div class="card-caption">Did the agent enforce the correct verdict?</div>', unsafe_allow_html=True)
        if "expected_verdict" in df.columns and "output_direction" in df.columns:
            vdf = df[["id","expected_verdict","output_direction"]].copy()
            vdf["Match"] = vdf.apply(lambda r: "✅ Correct" if str(r["expected_verdict"]).upper()==str(r["output_direction"]).upper() else "❌ Wrong", axis=1)
            vdf.columns = ["ID","Expected","Actual","Match"]
            st.dataframe(vdf, use_container_width=True, hide_index=True)
        else:
            st.caption("Add `expected_verdict` to test_cases.csv to enable.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Inspector</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select case:", df["id"].tolist(),
        format_func=lambda x: f"Case {x}  ·  {df[df['id']==x]['input'].values[0][:60]}…")
    r = df[df["id"] == sel].iloc[0]
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(f"""<div class="inspector-box">
          <div class="inspector-label">Input Prompt</div>
          <div style="font-style:italic;margin-bottom:1rem;">"{r.get('input','—')}"</div>
          <div class="inspector-label">Agent Output</div>
          <div style="font-size:0.82rem;line-height:1.55;">{r.get('output','—')}</div>
        </div>""", unsafe_allow_html=True)

    with right:
        cons  = r.get("binary_consistency", -1)
        faith = r.get("faithfulness_score", -1)
        hall  = r.get("hallucination_binary", -1)
        fc    = EMERALD if faith >= 0.8 else AMBER if faith >= 0.5 else ROSE
        fix   = r.get("policy_fix_suggestion", "")
        fix_html = (f"<div class='inspector-label' style='margin-top:0.75rem;'>Suggested Fix</div>"
                    f"<div style='font-size:0.82rem;font-style:italic;color:{AMBER};line-height:1.5;margin-bottom:0.75rem;'"
                    f">💡 {fix}</div>") if fix else ""
        st.markdown(f"""<div class="inspector-box">
          <div class="inspector-label">Judge Reasoning</div>
          <div style="font-size:0.82rem;line-height:1.55;margin-bottom:0.5rem;">{r.get('reasoning','Not recorded.')}</div>
          {fix_html}
          <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.75rem;">
            <span class="badge {'badge-pass' if cons==1 else 'badge-fail'}">Consistency: {cons}</span>
            <span class="badge {'badge-fail' if hall==1 else 'badge-ok'}">Halluc: {'YES' if hall==1 else 'NO'}</span>
            <span class="badge" style="background:{fc}18;color:{fc};border:1px solid {fc}40;">Faith: {faith:.2f}</span>
          </div>
          <div style="font-size:0.75rem;color:{SLATE_5};">
            Violation: <strong>{fmt_violation(r.get('policy_violation','—'))}</strong> &nbsp;·&nbsp;
            Tool: <strong>{r.get('tool_status','—')}</strong> &nbsp;·&nbsp;
            Model: <strong>{r.get('ka2l_model','—')}</strong>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KA2L ROUTING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""<div class="card" style="margin-bottom:1.5rem;">
      <div class="card-title">KA2L Pre-Inference Router</div>
      <div class="card-caption" style="margin-bottom:0;">Every prompt passes through <strong>GPT-2 locally on the backend</strong>
      before reaching Groq. The last-token hidden state is L2-normalised and its variance used as a
      <strong>semantic entropy score</strong>. Low entropy → fast 8b model. High entropy → powerful 70b model.</div>
    </div>""", unsafe_allow_html=True)

    if df.empty or "ka2l_destination" not in df.columns:
        st.info("No routing data. Trigger an eval run from the sidebar.")
    else:
        easy = int((df["ka2l_destination"] == "agent_easy").sum())
        hard = int((df["ka2l_destination"] == "agent_hard").sum())
        prechk = int((df["ka2l_destination"] == "precheck").sum())
        routed = easy + hard
        avg_var = df[df["ka2l_destination"] != "precheck"]["ka2l_variance"].replace(-1, float("nan")).mean()
        THRESHOLD = 0.5

        st.markdown(f"""<div class="kpi-grid">
          {kpi("8b-instant (easy)", str(easy), f"{(easy/len(df))*100:.0f}% of prompts", "indigo")}
          {kpi("70b-versatile (hard)", str(hard), f"{(hard/len(df))*100:.0f}% of prompts", "amber")}
          {kpi("Pre-checked", str(prechk), "blocked before LLM", "rose" if prechk else "emerald")}
          {kpi("Avg Entropy σ²", f"{avg_var:.5f}" if not pd.isna(avg_var) else "—", "routed cases only", "slate")}
          {kpi("Cost Saving", f"{((easy+prechk)/len(df))*100:.0f}%", "skipped 70b or LLM entirely", "emerald")}
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Routing Decisions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        COLOR_MAP = {"agent_easy": INDIGO, "agent_hard": AMBER, "precheck": ROSE, "unknown": SLATE_5}
        LABEL_MAP = {"agent_easy": "8b", "agent_hard": "70b", "precheck": "⛔", "unknown": "?"}

        with c1:
            st.markdown('<div class="card"><div class="card-title">Model Selected per Case</div><div class="card-caption">Indigo = 8b · Amber = 70b · Red = pre-checked (no LLM)</div>', unsafe_allow_html=True)
            rdf = df[["id","ka2l_destination","ka2l_variance","input"]].copy()
            rdf["color"] = rdf["ka2l_destination"].map(COLOR_MAP).fillna(SLATE_5)
            rdf["label"] = rdf["ka2l_destination"].map(LABEL_MAP).fillna("?")
            fig3 = go.Figure()
            for _, row in rdf.iterrows():
                fig3.add_trace(go.Bar(x=[f"Case {row['id']}"], y=[1], marker_color=row["color"],
                    text=row["label"], textposition="inside", textfont=dict(size=13, color="white"),
                    hovertext=str(row["input"])[:55], showlegend=False))
            fig3.update_layout(**CHART_LAYOUT, height=240, yaxis={**BASE_YAXIS, "visible": False}, bargap=0.35)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown(f'<div class="card"><div class="card-title">Semantic Entropy per Case</div><div class="card-caption">Below {THRESHOLD} = agent_easy · above = agent_hard · no bar = pre-checked</div>', unsafe_allow_html=True)
            vdf = df[df["ka2l_destination"] != "precheck"][["id","ka2l_variance"]].copy()
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=[f"Case {i}" for i in vdf["id"]], y=vdf["ka2l_variance"].replace(-1, None),
                mode="lines+markers", marker=dict(size=10, color=INDIGO, line=dict(width=2, color=BG_CARD)),
                line=dict(width=2.5, color=INDIGO),
                hovertemplate="Case %{x}<br>σ²: %{y:.6f}<extra></extra>"))
            fig4.add_hline(y=THRESHOLD, line_dash="dot", line_color=ROSE,
                annotation_text=f"threshold {THRESHOLD}", annotation_font_color=ROSE, annotation_position="top right")
            fig4.update_layout(**CHART_LAYOUT, height=240, yaxis={**BASE_YAXIS, "title": "σ²"})
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Routing Log</div>', unsafe_allow_html=True)
        log = [{"Case": f"Case {r['id']}", "Prompt": str(r.get("input",""))[:60]+"…",
                "Entropy": f"{r.get('ka2l_variance',-1):.6f}" if r.get("ka2l_destination") != "precheck" else "precheck",
                "Route": {"agent_easy":"🟢 easy","agent_hard":"🟡 hard","precheck":"⛔ blocked","unknown":"❓"}.get(r.get("ka2l_destination",""), "?"),
                "Model": r.get("ka2l_model","—"),
                "✓": "✅" if r.get("binary_consistency")==1 else "❌",
                "Faith.": f"{r.get('faithfulness_score','—'):.2f}" if isinstance(r.get("faithfulness_score"), float) else "—"}
               for _, r in df.iterrows()]
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="card" style="margin-bottom:1.5rem;"><div class="card-title">Agent Playground</div><div class="card-caption">Send a live prompt through the Railway backend — policy pre-check → KA2L routing → Groq agent.</div></div>', unsafe_allow_html=True)

    with st.expander("View current policy"):
        st.code(load_policy(), language="markdown")

    ci, cb = st.columns([4, 1])
    with ci:
        user_input = st.text_input("", placeholder="Try: Change my flight TKT-123 to next Tuesday", label_visibility="collapsed")
    with cb:
        run = st.button("Run →", use_container_width=True)

    if run and user_input:
        with st.spinner("Calling Railway backend…"):
            try:
                result = run_agent_api(user_input)
                r = result["routing"]
                dest = r.get("route", "unknown")
                precheck = result.get("precheck_triggered", False)

                ca, cb2 = st.columns([1.3, 1], gap="large")
                with ca:
                    st.markdown(f"""<div class="inspector-box">
                      <div class="inspector-label">Agent Response</div>
                      <div style="font-size:0.88rem;line-height:1.6;">{result["output"]}</div>
                    </div>""", unsafe_allow_html=True)
                with cb2:
                    if precheck:
                        rule = result.get("precheck_rule","precheck")
                        st.markdown(f"""<div class="inspector-box">
                          <div class="inspector-label">Policy Pre-Check</div>
                          <div style="margin-bottom:0.8rem;"><span class="badge badge-fail" style="font-size:0.8rem;padding:4px 14px;">⛔ Blocked</span></div>
                          <div style="font-size:0.82rem;color:{SLATE_3};line-height:1.8;">
                            <strong>Rule:</strong> {rule.replace("precheck_","").replace("_"," ")}<br>
                            <strong>LLM called:</strong> No<br>
                            <strong>Policy hash:</strong> <code>{result.get('policy_hash','—')}</code>
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        bc  = "badge-easy" if dest == "agent_easy" else "badge-hard"
                        mdl = "llama-3.1-8b-instant" if dest == "agent_easy" else "llama-3.3-70b-versatile"
                        st.markdown(f"""<div class="inspector-box">
                          <div class="inspector-label">KA2L Decision</div>
                          <div style="margin-bottom:0.8rem;"><span class="badge {bc}" style="font-size:0.8rem;padding:4px 14px;">{dest}</span></div>
                          <div style="font-size:0.82rem;color:{SLATE_3};line-height:1.8;">
                            <strong>Model:</strong> {mdl}<br>
                            <strong>Entropy σ²:</strong> {r.get('variance',0):.6f}<br>
                            <strong>Threshold:</strong> {r.get('threshold',0.5)}<br>
                            <strong>Policy hash:</strong> <code>{result.get('policy_hash','—')}</code>
                          </div>
                        </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Backend error: {e}")
    elif run:
        st.warning("Enter a prompt first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — POLICY EDITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="card" style="margin-bottom:1.5rem;"><div class="card-title">Policy Editor</div><div class="card-caption">Select a failed case, diagnose the failure, and append a new rule via the backend API.</div></div>', unsafe_allow_html=True)

    if df.empty:
        st.info("Trigger an eval run from the sidebar first.")
    else:
        failed = df[(df["binary_consistency"]==0) | (df["policy_violation"].notna() & (df["policy_violation"]!="None"))]

        if failed.empty:
            st.success("All cases passed — nothing to fix.")
            fr, suggestion = None, ""
        else:
            sid = st.selectbox("Select case:", failed["id"].tolist(),
                format_func=lambda x: f"Case {x}  ·  {failed[failed['id']==x]['input'].values[0][:45]}…")
            fr = failed[failed["id"]==sid].iloc[0]
            suggestion = fr.get("policy_fix_suggestion", "") or ""

        cl, cr = st.columns([1, 1.4], gap="large")

        with cl:
            st.markdown("**Failed Cases**")
            if fr is None:
                st.success("All cases passed — nothing to fix.")
            else:
                fix_block = (
                    f"<div class='inspector-label' style='margin-top:0.75rem;'>💡 Suggested Fix</div>"
                    f"<div style='font-size:0.8rem;font-style:italic;color:{AMBER};line-height:1.5;"
                    f"border-left:3px solid {AMBER};padding-left:0.6rem;margin-bottom:0.75rem;'>{suggestion}</div>"
                ) if suggestion else ""
                st.markdown(f"""<div class="inspector-box" style="margin-top:0.5rem;">
                  <div class="inspector-label">Input</div>
                  <div style="font-style:italic;margin-bottom:0.75rem;">"{fr['input']}"</div>
                  <div class="inspector-label">Agent Said</div>
                  <div style="font-size:0.8rem;margin-bottom:0.75rem;">{str(fr['output'])[:200]}…</div>
                  <div class="inspector-label">Judge Reasoning</div>
                  <div style="font-size:0.8rem;margin-bottom:0.75rem;">{fr.get('reasoning','Not recorded')}</div>
                  {fix_block}
                  <div style="font-size:0.75rem;color:{SLATE_5};">Violation: <strong>{fmt_violation(fr.get('policy_violation','—'))}</strong></div>
                </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown("**Write a Fix**")
            if suggestion:
                st.caption("✨ Pre-filled from judge suggestion — edit as needed.")
            author   = st.text_input("Author / initials", value="analyst", key="author")
            new_rule = st.text_area("New policy rule", height=110, value=suggestion,
                placeholder="e.g., Basic Economy tickets are non-refundable and cannot be cancelled.", key="new_rule")
            if st.button("Append to Policy"):
                if new_rule.strip():
                    try:
                        resp = append_to_policy_api(new_rule, author)
                        st.success(f"Policy updated on backend. Backup: `{resp.get('backup','—')}`")
                        st.info("Use the sidebar to trigger a new eval run, then check the Regression tab.")
                        load_policy.clear()
                    except Exception as e:
                        st.error(f"Failed: {e}")
                else:
                    st.warning("Write a rule first.")
            st.markdown("---")
            st.markdown("**Current Policy**")
            st.text_area("", value=load_policy(), height=260, disabled=True, key="pview", label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REGRESSION TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="card" style="margin-bottom:1.5rem;"><div class="card-title">Regression Tracker</div><div class="card-caption">Snapshot current results as a baseline, trigger a new eval run after a policy fix, then compare.</div></div>', unsafe_allow_html=True)

    if "baseline_snapshot" not in st.session_state:
        st.session_state["baseline_snapshot"] = None

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("**Baseline Snapshot**")
        if st.button("📸 Save current results as baseline"):
            if not df.empty:
                st.session_state["baseline_snapshot"] = df.copy()
                st.success("Baseline saved for this session.")
            else:
                st.warning("No results to snapshot.")
        if st.session_state["baseline_snapshot"] is not None:
            snap = st.session_state["baseline_snapshot"]
            st.caption(f"{len(snap)} cases · run_id: {snap['run_id'].iloc[0] if 'run_id' in snap.columns else '—'}")

    with col_b:
        st.markdown("**Current Run**")
        st.caption(f"{len(df)} cases · run_id: {df['run_id'].iloc[0] if 'run_id' in df.columns and not df.empty else '—'}")

    if st.session_state["baseline_snapshot"] is not None and not df.empty:
        if st.button("Compare →"):
            df_base = st.session_state["baseline_snapshot"]
            df_curr = df
            merged = df_base[["id","binary_consistency","faithfulness_score"]].merge(
                df_curr[["id","binary_consistency","faithfulness_score"]], on="id", suffixes=("_before","_after"))
            merged["Δ Consistency"]  = merged["binary_consistency_after"]  - merged["binary_consistency_before"]
            merged["Δ Faithfulness"] = merged["faithfulness_score_after"]  - merged["faithfulness_score_before"]

            reg = merged[merged["Δ Consistency"] < 0]
            imp = merged[merged["Δ Consistency"] > 0]
            if not reg.empty: st.error(f"⚠️  {len(reg)} regression(s) — some cases got worse.")
            if not imp.empty: st.success(f"✅  {len(imp)} improvement(s) — these cases got better.")
            if reg.empty and imp.empty: st.info("No change in consistency scores between runs.")

            st.markdown('<div class="section-header">Delta Table</div>', unsafe_allow_html=True)

            def hl(val):
                try:
                    v = float(val)
                    if v > 0: return f"background-color:{EMERALD}18;color:{EMERALD}"
                    if v < 0: return f"background-color:{ROSE}15;color:{ROSE}"
                except: pass
                return ""

            disp = merged.rename(columns={"id":"Case","binary_consistency_before":"Consist. Before",
                "binary_consistency_after":"Consist. After","faithfulness_score_before":"Faith. Before",
                "faithfulness_score_after":"Faith. After"})
            st.dataframe(disp.style.applymap(hl, subset=["Δ Consistency","Δ Faithfulness"]),
                use_container_width=True, hide_index=True)

            st.markdown('<div class="section-header">Faithfulness Trend</div>', unsafe_allow_html=True)
            cases = [f"Case {i}" for i in merged["id"]]
            fig5  = go.Figure()
            fig5.add_trace(go.Scatter(x=cases, y=merged["faithfulness_score_before"], name="Before",
                mode="lines+markers", line=dict(color=SLATE_5, dash="dash", width=2), marker=dict(size=8, color=SLATE_5)))
            fig5.add_trace(go.Scatter(x=cases, y=merged["faithfulness_score_after"], name="After",
                mode="lines+markers", line=dict(color=INDIGO, width=2.5),
                marker=dict(size=9, color=INDIGO, line=dict(width=2, color=BG_CARD))))
            fig5.update_layout(**CHART_LAYOUT, height=260,
                yaxis={**BASE_YAXIS, "range": [0,1.15], "title": "Faithfulness"},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Save a baseline snapshot first, then trigger a new eval run from the sidebar and click Compare.")
