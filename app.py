import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models import RunRecord
from src.pipeline import run_evaluation
from src.dashboard import render_ui


st.set_page_config(
    page_title="LLM-as-a-Judge Evaluator",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("LLM-as-a-Judge Evaluator Dashboard")
st.markdown(
    "Evaluate your sandbox's code execution outputs using an LLM judge "
    "with relevance, hallucination, and toxicity scoring."
)

with st.sidebar:
    st.header("Controls")

    dataset_path = st.text_input("Dataset path", value="data/golden_dataset.json")

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running evaluation pipeline..."):
            try:
                records = run_evaluation(dataset_path)
                st.session_state["records"] = [r.model_dump() for r in records]
                st.success(f"Evaluation complete: {len(records)} samples")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    uploaded_file = st.file_uploader("Or load results JSON", type="json")
    if uploaded_file:
        data = json.loads(uploaded_file.read())
        st.session_state["records"] = data
        st.success(f"Loaded {len(data)} records")

    if st.button("Clear Results"):
        if "records" in st.session_state:
            del st.session_state["records"]
        st.rerun()

if "records" not in st.session_state or not st.session_state["records"]:
    st.info("Run an evaluation or upload results to see metrics.")
    st.stop()

render_ui(st.session_state["records"])
