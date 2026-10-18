"""Streamlit rendering helpers (layout, charts, cards). Logic lives in components.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import EMOTIONS, HELPLINES, NEGATIVE, PCOLS

EMOTION_COLORS = {
    "Joy": "#f2b134", "Sadness": "#4c78a8", "Anger": "#d1495b", "Fear": "#7d5ba6",
    "Anxiety": "#e07b39", "Stress": "#b5524a", "Surprise": "#3fa7a3", "Neutral": "#9aa0a6",
}


def render_header():
    st.title("Mood Journal")
    st.caption("Emotion-aware journaling that tracks how your mood changes over time.")


def render_disclaimer():
    st.info(
        "**This is not a diagnostic tool.** It estimates emotions from the words you write and cannot tell you "
        "whether you have any mental-health condition. It does not replace a doctor, therapist or counsellor. "
        "**It cannot respond to emergencies** - if you are in immediate danger, contact local emergency services "
        "or a crisis helpline.", icon="ℹ️")


def render_consent() -> bool:
    with st.expander("Privacy & consent", expanded=True):
        st.markdown(
            "- Only text you **choose to submit** is analysed.\n"
            "- Your raw text is **not stored** - only the emotion scores and a timestamp.\n"
            "- You can **export or delete** your data any time from the sidebar.\n"
            "- You can withdraw consent by un-ticking the box below.")
        return st.checkbox("I understand and agree to have my entries analysed.", key="consent")


def render_entry_result(probs: np.ndarray, model_label: str):
    order = np.argsort(probs)[::-1]
    top = EMOTIONS[order[0]]
    st.markdown(f"#### Detected: **{top}** ({probs[order[0]]:.0%})")
    fig = go.Figure(go.Bar(x=[probs[i] for i in order][::-1], y=[EMOTIONS[i] for i in order][::-1], orientation="h",
                           marker_color=[EMOTION_COLORS[EMOTIONS[i]] for i in order][::-1],
                           text=[f"{probs[i]:.0%}" for i in order][::-1], textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 1.1], visible=False),
                      showlegend=False)
    st.plotly_chart(fig)
    if probs.max() < 0.45:
        st.caption("The model is not very confident about this entry - emotions in short or ambiguous text are hard to read.")
    st.caption(f"Model: {model_label}. Sarcasm and very short text are known weak spots.")


def render_trends(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame, tau: float, flags: np.ndarray, view: str):
    data = {"Daily": daily, "Weekly": weekly, "Monthly": monthly}[view]
    if data.empty:
        st.write("No data for this view yet.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["S"], mode="lines+markers", name="Negative affect",
                             line=dict(color="#9aa0a6", width=1.5), opacity=0.8))
    if view == "Daily":
        fig.add_trace(go.Scatter(x=data.index, y=data["S_ema"], mode="lines", name="Smoothed trend",
                                 line=dict(color="#1f3b73", width=3)))
        fig.add_hline(y=tau, line_dash="dot", line_color="crimson", annotation_text="support threshold")
        for ts in data.index[flags]:
            fig.add_vrect(x0=ts - pd.Timedelta(hours=12), x1=ts + pd.Timedelta(hours=12),
                          fillcolor="crimson", opacity=0.10, line_width=0)
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(range=[0, 1], title="negative affect (0-1)"),
                      title=f"{view} negative-affect score", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig)

    area = go.Figure()
    for e in EMOTIONS:
        area.add_trace(go.Scatter(x=data.index, y=data[f"p_{e}"], mode="lines", stackgroup="one", name=e,
                                  line=dict(width=0.5, color=EMOTION_COLORS[e])))
    area.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), title="Which emotions contribute",
                       yaxis=dict(range=[0, 1], title="share"), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(area)

    t = data[["dominant", "S"]].copy()
    t.index = t.index.strftime("%Y-%m-%d")
    st.dataframe(t.rename(columns={"dominant": "dominant emotion", "S": "negative affect"}).round(2).tail(14))


def render_recommendation(rec: dict):
    if rec["level"] == "escalation":
        st.error(f"**{rec['title']}**\n\n{rec['text']}", icon="💙")
        st.markdown("**Support lines** *(please check these are right for your region)*")
        for name, number in HELPLINES.items():
            st.markdown(f"- {name}: **{number}**")
    elif rec["level"] == "suggestion":
        st.warning(f"**{rec['title']}**\n\n{rec['text']}", icon="🌿")
    else:
        st.success(f"**{rec['title']}**\n\n{rec['text']}", icon="✅")


def render_summary_metrics(daily: pd.DataFrame, n_entries: int):
    c1, c2, c3 = st.columns(3)
    c1.metric("Entries analysed", n_entries)
    c2.metric("Days with entries", len(daily))
    if len(daily):
        c3.metric("Latest smoothed negative affect", f"{daily['S_ema'].iloc[-1]:.2f}")
