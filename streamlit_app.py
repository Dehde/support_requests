# streamlit_app.py

"""
Streamlit App that:
- Fetches Langfuse traces (filtered by date & tag)
- Builds a DataFrame with user question, conversation history, model thoughts, etc.
- Shows them in a UI
- Allows editing an 'Expected Answer' field
- Offers a dropdown to pick or create a dataset
- Uses the Python SDK to create/update (upsert) a single dataset item
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


def create_data_rows(traces: List[LangfuseTrace], trace_scores_map: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
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
    (We assume the direct REST call is the simplest approach here.)
    """
    resp = requests.get(
        f"{LANGFUSE_HOST}/api/public/datasets",
        auth=HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_or_create_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Look for a dataset by name, create it if not found, and return a dict with at least {'id', 'name'}.
    """
    # 1) Check if dataset exists
    existing = [ds for ds in list_datasets() if ds.get("name") == dataset_name]
    if existing:
        return existing[0]

    # 2) Create a new dataset using the python client
    client = init_langfuse()
    created = client.datasets.create(name=dataset_name)
    return {"id": created["id"], "name": created["name"]}


def create_or_update_dataset_item(
    dataset_name: str,
    item_id: str,
    input_data: Dict[str, Any],
    expected_answer: str
) -> None:
    """
    Use the Python SDK's create_dataset_item() to create or update a dataset item.
    By specifying the same 'id', future calls will overwrite the same dataset item.
    """
    client = init_langfuse()
    # This assumes create_dataset_item() will do an upsert if 'id' already exists.
    client.create_dataset_item(
        dataset_name=dataset_name,
        id=item_id,  # item identifier
        input=input_data,
        expected_output={"answer": expected_answer},
        metadata={}  # optionally add any extra metadata here
    )


def main():
    st.title("Langfuse Trace Reviewer (SDK Upsert)")

    # Load trace data into DataFrame
    df = load_traces_data()
    if df.empty:
        st.warning("No traces found matching the filter!")
        return

    # List existing datasets
    st.subheader("Pick an existing dataset or create a new one")
    datasets = list_datasets()
    dataset_names = [ds["name"] for ds in datasets] if datasets else []

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
            help="Enter a dataset name to create a new one."
        )

    # Decide which dataset name is being used
    chosen_dataset_name = new_dataset_name.strip() if new_dataset_name.strip() else None
    if not chosen_dataset_name and selected_dataset_name != "<None>":
        chosen_dataset_name = selected_dataset_name

    st.subheader("Select a Trace to Label")
    trace_ids = df["ID"].tolist()
    selected_trace_id = st.selectbox(
        "Select a trace to review",
        options=trace_ids,
        format_func=lambda x: f"Trace: {x}"
    )

    trace_data = df[df["ID"] == selected_trace_id].iloc[0]

    st.header("Inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("User Question")
        st.text_area("", value=trace_data["User Question"], height=200, disabled=True)
    with c2:
        st.subheader("Conversation History")
        st.text_area("", value=trace_data["Conversation History"], height=200, disabled=True)

    st.header("Retrieved Context")
    st.text_area("", value=trace_data["Retrieved Context"], height=200, disabled=True)

    st.header("Model Output")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Model Thoughts")
        st.text_area("", value=trace_data["Model Thoughts"], height=200, disabled=True)
    with c4:
        st.subheader("Answer")
        st.text_area("", value=trace_data["Answer"], height=200, disabled=True)

    st.header("Expected Answer")
    expected_answer = st.text_area(
        "",
        value=trace_data["Expected Answer"],
        height=200,
        key=f"expected_answer_{selected_trace_id}"
    )

    if st.button("Save Dataset Item"):
        # Ensure a dataset name has been chosen/entered
        if not chosen_dataset_name:
            st.error("Please pick an existing dataset or enter a new dataset name.")
            return

        # Update the local DataFrame
        df.loc[df["ID"] == selected_trace_id, "Expected Answer"] = expected_answer
        export_to_csv(df, OUTPUT_FILE)  # optional local record

        # Make sure the dataset exists (create if needed)
        dataset_obj = get_or_create_dataset(chosen_dataset_name)

        # Prepare the input
        input_data = {
            "user_question": trace_data["User Question"],
            "conversation_history": trace_data["Conversation History"],
            "context": trace_data["Retrieved Context"],
        }

        # Upsert the single item to the dataset
        create_or_update_dataset_item(
            dataset_name=dataset_obj["name"],
            item_id=selected_trace_id,  # Reuse trace ID as dataset item ID
            input_data=input_data,
            expected_answer=expected_answer
        )

        st.success(
            f"Successfully upserted trace '{selected_trace_id}' "
            f"into dataset '{dataset_obj['name']}' "
            f"with updated Expected Answer."
        )


if __name__ == "__main__":
    main()
