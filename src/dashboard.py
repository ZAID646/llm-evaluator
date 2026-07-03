import pandas as pd
import plotly.express as px
import streamlit as st


def render_ui(records_data: list[dict]):
    df = pd.DataFrame(records_data)

    tab1, tab2, tab3 = st.tabs(["Summary", "Per-Sample Breakdown", "Raw Data"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        total = len(df)
        passed = df["passed"].sum()
        fail = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        col1.metric("Total Samples", total)
        col2.metric("Passed", int(passed))
        col3.metric("Failed", int(fail))
        col4.metric("Pass Rate", f"{pass_rate:.1f}%")

        st.divider()

        score_cols = ["relevance", "hallucination", "toxicity"]
        available = [c for c in score_cols if c in df.columns and df[c].notna().any()]

        if available:
            avg_scores = df[available].mean()
            st.subheader("Average Scores")
            score_df = pd.DataFrame({
                "Criterion": [c.capitalize() for c in available],
                "Average Score": [avg_scores[c] for c in available],
            })
            fig = px.bar(
                score_df,
                x="Criterion",
                y="Average Score",
                range_y=[0, 5],
                text_auto=".2f",
                color="Average Score",
                color_continuous_scale="viridis",
            )
            st.plotly_chart(fig, use_container_width=True)

        if "target_latency_ms" in df.columns:
            st.subheader("Target Latency Distribution")
            lat_df = df[df["target_latency_ms"].notna()]
            if not lat_df.empty:
                fig2 = px.histogram(
                    lat_df,
                    x="target_latency_ms",
                    nbins=20,
                    labels={"target_latency_ms": "Latency (ms)"},
                )
                st.plotly_chart(fig2, use_container_width=True)

        if "judge_tokens" in df.columns:
            total_judge_tokens = int(df["judge_tokens"].sum())
            total_target_tokens = int(df["target_tokens"].sum()) if "target_tokens" in df.columns else 0
            total_tokens = total_judge_tokens + total_target_tokens
            cost_estimate = total_tokens * 0.000002

            st.subheader("Token Usage & Cost")
            tok_col1, tok_col2, tok_col3, tok_col4 = st.columns(4)
            tok_col1.metric("Target Tokens", f"{total_target_tokens:,}")
            tok_col2.metric("Judge Tokens", f"{total_judge_tokens:,}")
            tok_col3.metric("Total Tokens", f"{total_tokens:,}")
            tok_col4.metric("Est. Cost", f"${cost_estimate:.6f}")

    with tab2:
        for _, row in df.iterrows():
            passed_icon = "PASS" if row["passed"] else "FAIL"
            with st.expander(
                f"[{passed_icon}] {row.get('sample_id', '?')}: {row['prompt'][:80]}...",
                expanded=not row["passed"],
            ):
                st.text_area("Prompt", row["prompt"], height=80, disabled=True)
                st.text_area("Expected Answer", row["expected_answer"], height=80, disabled=True)

                if row.get("target_output"):
                    st.text_area("Target Output", row["target_output"], height=100, disabled=True)
                if row.get("target_error"):
                    st.error(f"Target Error: {row['target_error']}")

                if row.get("relevance") is not None:
                    r, h, t = row["relevance"], row["hallucination"], row["toxicity"]
                    st.metric("Relevance", f"{r}/5")
                    st.metric("Hallucination (5=best)", f"{h}/5")
                    st.metric("Toxicity (5=best)", f"{t}/5")

                if row.get("reasoning"):
                    st.caption(f"Judge reasoning: {row['reasoning']}")

                st.caption(
                    f"Target latency: {row.get('target_latency_ms', 'N/A')}ms | "
                    f"Judge latency: {row.get('judge_latency_ms', 'N/A')}ms | "
                    f"Target tokens: {row.get('target_tokens', 'N/A')} | "
                    f"Judge tokens: {row.get('judge_tokens', 'N/A')}"
                )

    with tab3:
        st.json(records_data)
