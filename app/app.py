"""Mood Journal - Streamlit app.   Run from the repo root:  streamlit run app/app.py

Needs: pip install streamlit plotly pandas numpy emoji nltk scikit-learn joblib   (+ torch transformers for a transformer model)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # so `import components, ui` works from anywhere

import streamlit as st

import components as C
import ui

st.set_page_config(page_title="Mood Journal", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    return C.Predictor()


def main():
    ui.render_header()
    ui.render_disclaimer()

    params = C.load_params()

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Your data")
        user_id = st.text_input("Profile name", value="me", help="Entries are kept per profile name on this machine.").strip() or "me"
        view = st.radio("Trend view", ["Daily", "Weekly", "Monthly"], horizontal=True)
        st.divider()
        hist = C.load_history(user_id)
        st.caption(f"{len(hist)} stored scores for '{user_id}'")
        st.download_button("Export my data (CSV)", C.export_history_csv(user_id), file_name=f"{user_id}_mood_scores.csv",
                           mime="text/csv")
        with st.expander("Delete my data"):
            sure = st.checkbox("Yes, permanently delete all my stored scores")
            if st.button("Delete", disabled=not sure):
                st.toast(f"Deleted {C.delete_history(user_id)} records")
                st.rerun()
        st.divider()
        with st.expander("Demo data (simulated)"):
            st.caption("Adds 4 weeks of **simulated** scores (random numbers, not real text) to a profile named 'demo'.")
            if st.button("Load demo profile"):
                C.delete_history("demo")
                C.add_demo_history("demo")
                st.toast("Demo profile created - type 'demo' as profile name")
        with st.expander("About the trend rule"):
            st.caption(f"Smoothing alpha={params['alpha']}; support message when the smoothed negative-affect score stays above "
                       f"{params['tau']} for {params['k']} or more entry-days. Parameters come from simulated timelines "
                       "(notebook 06) - not clinically validated.")

    consent = ui.render_consent()

    # ------------------------------------------------------------ new entry
    left, right = st.columns([1, 1])
    with left:
        st.subheader("How are you feeling?")
        with st.form("entry", clear_on_submit=True):
            text = st.text_area("Write a few sentences", height=160, placeholder="e.g. I couldn't sleep, the deadline is stressing me out...")
            submitted = st.form_submit_button("Analyse", disabled=not consent)
        if not consent:
            st.caption("Tick the consent box above to enable analysis.")
        if submitted:
            if len(text.split()) < 3:
                st.warning("Please write at least a few words - very short text is hard to read reliably.")
            else:
                try:
                    pred = get_predictor()
                except FileNotFoundError as e:
                    st.error(str(e))
                else:
                    probs = pred.predict_proba(text)
                    C.add_prediction(user_id, probs)          # only scores + timestamp are stored, never the text
                    st.session_state["last"] = (probs, pred.label)

        if "last" in st.session_state:
            ui.render_entry_result(*st.session_state["last"])

    hist = C.load_history(user_id)
    daily = C.daily_summary(hist, params["alpha"]) if len(hist) else None

    with right:
        st.subheader("Suggestions")
        if daily is None or daily.empty:
            st.write("Add a few entries over several days to see trends and suggestions.")
        else:
            flags = C.escalation_flags(daily, params["tau"], params["k"], params["max_gap_days"])
            ui.render_summary_metrics(daily, len(hist))
            ui.render_recommendation(C.recommend(daily, flags, params["tau"]))

    # ------------------------------------------------------------ trends
    st.divider()
    st.subheader("Your mood over time")
    if daily is None or daily.empty:
        st.write("Nothing to show yet.")
    else:
        ui.render_trends(daily, C.window_summary(hist, "7D"), C.window_summary(hist, "MS"),
                         params["tau"], flags, view)

    st.caption("Mood Journal is a wellness-support tool, not a medical device. Trends reflect the words you wrote, "
               "not your clinical state.")


main()
