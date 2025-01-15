# streamlit_app.py

"""
Streamlit App that:
- Fetches Langfuse traces (filtered by date & tag)
- Builds a DataFrame with user question, conversation history, model thoughts, etc.
- Shows them in a UI
- Allows editing an 'Expected Answer' field
- Offers a dropdown to pick or create a dataset
- Sends an upsert request for the selected trace to Langfuse
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
    """
    Initialize the Langfuse client using secrets.
    """
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )


def fetch_filtered_traces(langfuse_client: Langfuse, tag: str, start_date: str) -> List[LangfuseTrace]:
    """
    Fetch all traces with given tag since 'start_date'.
    Returns them as a list of LangfuseTrace dataclasses.
    """
    start_dt = pd.to_datetime(start_date, utc=True)

    traces_response = langfuse_client.fetch_traces(tags=[tag])
    all_traces = traces_response.data  # typically a list of "Trace" objects from the client

    filtered = []
    for t in all_traces:
        trace_dt_utc = pd.to_datetime(t.timestamp, utc=True)
        if trace_dt_utc >= start_dt:
            filtered.append(
                LangfuseTrace(
                    id=t.id,
                    name=t.name,
                    tags=t.tags or [],
                    timestamp=trace_dt_utc,
                    metadata=t.metadata or {},
                    output=t.output
                )
            )
    return filtered


def request_scores() -> List[Dict[str, Any]]:
    """
    Request all scores from Langfuse via public API.
    """
    response = requests.get(
        f"{LANGFUSE_HOST}/api/public/scores",
        auth=HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json().get("data", [])


def build_trace_scores_map(scores: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a map trace_id -> list of score dicts
    """
    trace_scores_map = defaultdict(list)
    for score in scores:
        trace_id = score.get("traceId")
        if trace_id:
            trace_scores_map[trace_id].append(score)
    return trace_scores_map


def create_data_rows(traces: List[LangfuseTrace], trace_scores_map: Dict[str, List[Dict[str, Any]]]) -> List[
    Dict[str, Any]]:
    """
    Create a list of row dicts for each trace, including conversation,
    retrieved contexts, and any relevant scores.
    """
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
            "Expected Answer": original_answer,  # For editing
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


@st.cache_data
def list_datasets() -> List[Dict[str, Any]]:
    """
    Return all existing datasets from Langfuse via public API.
    (We use a direct REST call here; you could also do `client.datasets.list()` if available.)
    """
    url = f"{LANGFUSE_HOST}/api/public/datasets"
    resp = requests.get(
        url,
        auth=HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_or_create_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Look for a dataset by name, create it if not found, and return the dataset object.
    """
    # 1) See if dataset already exists
    existing = [ds for ds in list_datasets() if ds.get("name") == dataset_name]
    if existing:
        return existing[0]

    # 2) If not found, create a new dataset
    client = init_langfuse()
    created = client.datasets.create(name=dataset_name)
    # The returned object from the python client might differ in shape.
    # But let's assume it returns something with at least 'id' and 'name'.
    return {
        "id": created["id"],
        "name": created["name"]
    }


def upsert_dataset_item(
        dataset_id: str,
        external_id: str,
        input_data: Dict[str, Any],
        expected_output: str
) -> None:
    """
    Create or update a single dataset item with the given externalId.
    If it doesn't exist, it will be created; otherwise updated.

    This uses a direct REST call to the upsert endpoint:
      PUT /api/public/datasets/{datasetId}/items
    """
    url = f"{LANGFUSE_HOST}/api/public/datasets/{dataset_id}/items"
    payload = {
        "externalId": external_id,
        "input": input_data,
        "expectedOutput": expected_output
    }
    resp = requests.put(
        url,
        auth=HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
        headers={"Content-Type": "application/json"},
        json=payload
    )
    resp.raise_for_status()


def main():
    st.title("Langfuse Trace Reviewer (Dataset Upsert)")

    df = load_traces_data()
    if df.empty:
        st.warning("No traces found matching the filter!")
        return

    st.subheader("Pick an existing dataset or create a new one")

    datasets = list_datasets()  # returns list of { 'id':..., 'name':... }
    dataset_names = [ds["name"] for ds in datasets] if datasets else []

    # Two ways: (A) pick from dropdown, (B) create new
    col_ds_left, col_ds_right = st.columns([2, 2])
    with col_ds_left:
        selected_dataset_name = st.selectbox(
            "Existing Datasets",
            options=["<None>"] + dataset_names,
            index=0,
            help="Pick an existing dataset name"
        )
    with col_ds_right:
        new_dataset_name = st.text_input(
            "Or create a new dataset",
            value="",
            help="Enter a new dataset name here if you want to create it."
        )

    # Decide which dataset name is being used
    # Priority: if user typed a new name, use that; else use dropdown
    chosen_dataset_name = new_dataset_name.strip() if new_dataset_name.strip() else None
    if not chosen_dataset_name and selected_dataset_name != "<None>":
        chosen_dataset_name = selected_dataset_name

    # 3) Let user pick a trace from a dropdown
    st.subheader("Select a Trace to Label")
    trace_ids = df["ID"].tolist()
    selected_trace_id = st.selectbox(
        "Select a trace to review",
        options=trace_ids,
        format_func=lambda x: f"Trace: {x}"
    )

    # 4) Display the chosen trace data
    trace_data = df[df["ID"] == selected_trace_id].iloc[0]

    st.header("Inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("User Question")
        st.text_area(label="", value=trace_data["User Question"], height=200, disabled=True)
    with c2:
        st.subheader("Conversation History")
        st.text_area(label="", value=trace_data["Conversation History"], height=200, disabled=True)

    st.header("Retrieved Context")
    st.text_area(label="", value=trace_data["Retrieved Context"], height=200, disabled=True)

    st.header("Model Output")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Model Thoughts")
        st.text_area(label="", value=trace_data["Model Thoughts"], height=200, disabled=True)
    with c4:
        st.subheader("Answer")
        st.text_area(label="", value=trace_data["Answer"], height=200, disabled=True)

    st.header("Expected Answer")
    expected_answer = st.text_area(
        label="",
        value=trace_data["Expected Answer"],
        height=200,
        key=f"expected_answer_{selected_trace_id}"
    )

    # 5) Save button to upsert this item to the chosen dataset
    if st.button("Save Dataset Item"):
        if not chosen_dataset_name:
            st.error("Please pick an existing dataset or enter a new dataset name.")
            return

        # Update local DataFrame with new Expected Answer
        df.loc[df["ID"] == selected_trace_id, "Expected Answer"] = expected_answer
        # Optionally export to CSV if you want a local record
        export_to_csv(df, OUTPUT_FILE)

        # 5a) Ensure the dataset actually exists (create if needed)
        dataset_obj = get_or_create_dataset(chosen_dataset_name)

        # 5b) Upsert this single item (the selected trace) via direct REST call
        input_data = {
            "user_question": trace_data["User Question"],
            "conversation_history": trace_data["Conversation History"],
            "context": trace_data["Retrieved Context"]
        }
        upsert_dataset_item(
            dataset_id=dataset_obj["id"],
            external_id=selected_trace_id,  # The trace ID as externalId
            input_data=input_data,
            expected_output=expected_answer
        )

        st.success(
            f"Upserted trace '{selected_trace_id}' into dataset '{dataset_obj['name']}' "
            f"with updated Expected Answer!"
        )


if __name__ == "__main__":
    main()
