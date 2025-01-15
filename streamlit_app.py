# streamlit_app.py

"""
Streamlit App that:
- Fetches Langfuse traces (filtered by date & tag)
- Builds a DataFrame with user question, conversation history, model thoughts, etc.
- Shows them in a UI
- Allows editing an 'Expected Answer' field
- Sends updated data to create a dataset in Langfuse
"""

import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any
from langfuse import Langfuse
from trace_model import LangfuseTrace
from collections import defaultdict


START_DATE = "2024-12-17"
TARGET_TAG = "app_id=d6bfd7f4-39a0-4824-8720-a8b79d32f586"
OUTPUT_FILE = "traces_processed.csv"

LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST = st.secrets["LANGFUSE_HOST"]


@st.cache_resource
def init_langfuse() -> Langfuse:
    return Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)


def fetch_filtered_traces(langfuse_client: Langfuse, tag: str, start_date: str) -> List[LangfuseTrace]:
    """
    Fetch all traces with given tag since 'start_date'.
    Returns them as a list of LangfuseTrace dataclasses.
    """
    start_dt = pd.to_datetime(start_date, utc=True)

    traces_response = langfuse_client.fetch_traces(tags=[tag])
    all_traces = traces_response.data  # this is typically a list of some "Trace" objects

    filtered = []
    for t in all_traces:
        trace_dt_utc = pd.to_datetime(t.timestamp, utc=True)
        if trace_dt_utc >= start_dt:
            filtered.append(LangfuseTrace(
                id=t.id,
                name=t.name,
                tags=t.tags or [],
                timestamp=trace_dt_utc,
                metadata=t.metadata or {},
                output=t.output
            ))
    return filtered


def request_scores() -> List[Dict[str, Any]]:
    response = requests.get(
        f"{LANGFUSE_HOST}/api/public/scores",
        auth=HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json().get("data", [])


def build_trace_scores_map(scores: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    trace_scores_map = defaultdict(list)
    for score in scores:
        trace_id = score.get("traceId")
        if trace_id:
            trace_scores_map[trace_id].append(score)
    return trace_scores_map


def create_data_rows(traces, trace_scores_map):
    rows = []
    allowed_scores = {"user_feedback", "HumanAnswerCorrectness", "HumanActionNeeded"}

    for trace in traces:
        timestamp_str = trace.timestamp.strftime("%Y-%m-%d %H:%M")

        # Extract relevant data from metadata
        user_question = trace.metadata.get("user_question", "")
        model_thoughts = trace.metadata.get("model_thoughts", "")

        # Safely handle conversation history, which might be None
        conversation_entries = trace.metadata.get("conversation_history") or []
        conversation_history = "\n".join(
            f"{entry.get('role', 'unknown')}: {entry.get('content', '')}"
            for entry in conversation_entries
        )

        # Combine retrieved contexts (top 5 by ascending cosine_distance) into multiline
        retrieved_contexts = trace.metadata.get("retrieved_contexts", [])
        retrieved_contexts_sorted = sorted(retrieved_contexts, key=lambda c: c.get("cosine_distance", 9999))[:5]
        context_text = ""
        for i, ctx in enumerate(retrieved_contexts_sorted, start=1):
            link = ctx.get("link", "No link")
            dist = ctx.get("cosine_distance", "N/A")
            context_text += f"[Context {i}]\nLink: {link}\nCosine Distance: {dist}\n\n"

        # The original answer is in trace.output
        original_answer = trace.output if trace.output else ""

        # Build row
        row = {
            "ID": trace.id,
            "Timestamp": timestamp_str,
            "User Question": user_question,
            "Conversation History": conversation_history,
            "Retrieved Context": context_text.strip(),
            "Model Thoughts": model_thoughts,
            "Answer": original_answer,
            # For editing
            "Expected Answer": original_answer,
            "Name": trace.name,
            "Tags": ", ".join(trace.tags) if trace.tags else "",
        }

        # Attach relevant scores
        scores_for_trace = trace_scores_map.get(trace.id, [])
        for sc in scores_for_trace:
            name = sc.get("name", "")
            if name in allowed_scores:
                data_type = sc.get("dataType", "")
                if data_type == "CATEGORICAL":
                    row[name] = sc.get("stringValue", "")
                else:
                    row[name] = sc.get("value", "")

        rows.append(row)

    return rows


def export_to_csv(df: pd.DataFrame, file_name: str = OUTPUT_FILE) -> None:
    """
    Utility to export the final DataFrame to CSV if you want a local record.
    """
    df.to_csv(file_name, index=False)


@st.cache_data
def load_traces_data() -> pd.DataFrame:
    """
    Fetch & build the DataFrame of traces (with relevant fields).
    """
    client = init_langfuse()
    traces = fetch_filtered_traces(client, TARGET_TAG, START_DATE)
    scores = request_scores()
    trace_scores_map = build_trace_scores_map(scores)
    rows = create_data_rows(traces, trace_scores_map)
    df = pd.DataFrame(rows)
    preferred_cols = [
        "ID", "Timestamp", "User Question", "Conversation History",
        "Retrieved Context", "Model Thoughts", "Answer", "Expected Answer",
        "Name", "Tags", "user_feedback", "HumanAnswerCorrectness", "HumanActionNeeded"
    ]
    existing_cols = [c for c in preferred_cols if c in df.columns]
    df = df.reindex(columns=existing_cols + [c for c in df.columns if c not in existing_cols])
    return df


def main():
    st.title("Langfuse Trace Reviewer (with direct init)")

    df = load_traces_data()
    if df.empty:
        st.warning("No traces found matching the filter!")
        return

    # Let user pick a trace from a dropdown
    trace_ids = df["ID"].tolist()
    selected_trace_id = st.selectbox(
        "Select a trace to review",
        options=trace_ids,
        format_func=lambda x: f"Trace: {x}"
    )

    # Show that row
    trace_data = df[df["ID"] == selected_trace_id].iloc[0]

    # Display inputs
    st.header("Inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("User Question")
        st.text_area(
            label="",
            value=trace_data["User Question"],
            height=200,
            disabled=True
        )
    with c2:
        st.subheader("Conversation History")
        st.text_area(
            label="",
            value=trace_data["Conversation History"],
            height=200,
            disabled=True
        )

    # Display context
    st.header("Retrieved Context")
    st.text_area(
        label="",
        value=trace_data["Retrieved Context"],
        height=200,
        disabled=True
    )

    # Display model output
    st.header("Model Output")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Model Thoughts")
        st.text_area(
            label="",
            value=trace_data["Model Thoughts"],
            height=200,
            disabled=True
        )
    with c4:
        st.subheader("Answer")
        st.text_area(
            label="",
            value=trace_data["Answer"],
            height=200,
            disabled=True
        )

    # Editable field
    st.header("Expected Answer")
    expected_answer = st.text_area(
        label="",
        value=trace_data["Expected Answer"],
        height=200,
        key=f"expected_answer_{selected_trace_id}"
    )

    # Dataset name + Save
    c5, c6 = st.columns([3,1])
    with c5:
        dataset_name = st.text_input("Dataset Name", value="")
    with c6:
        if st.button("Save to Langfuse"):
            if not dataset_name.strip():
                st.error("Please enter a dataset name.")
                return

            # Update local data with new expected answer
            df.loc[df["ID"] == selected_trace_id, "Expected Answer"] = expected_answer

            # Export to CSV if you want to keep track locally (optional)
            export_to_csv(df, OUTPUT_FILE)

            # Prepare dataset items from entire df
            dataset_items = []
            for _, row in df.iterrows():
                dataset_items.append({
                    "input": {
                        "user_question": row["User Question"],
                        "conversation_history": row["Conversation History"],
                        "context": row["Retrieved Context"]
                    },
                    "expected_output": row["Expected Answer"]
                })

            # Create dataset in Langfuse
            client = init_langfuse()
            client.datasets.create(
                name=dataset_name.strip(),
                items=dataset_items
            )
            st.success(f"Dataset '{dataset_name}' created successfully in Langfuse!")


if __name__ == "__main__":
    main()
