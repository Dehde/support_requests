# streamlit_app.py

import streamlit as st
import pandas as pd

from langfuse_utils import LangfuseClient, OUTPUT_FILE
from trace_model import LangfuseTrace

LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST = st.secrets["LANGFUSE_HOST"]


@st.cache_resource
def get_langfuse_client() -> LangfuseClient:
    return LangfuseClient(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )


@st.cache_data
def load_data(_client: LangfuseClient) -> pd.DataFrame:
    """
    Fetch the DataFrame of traces from the LangfuseClient.
    The leading underscore in _client tells Streamlit not to hash it.
    """
    return _client.load_traces_as_dataframe()


@st.cache_data
def load_active_score_configs(_client: LangfuseClient) -> pd.DataFrame:
    """
    Fetch all score configs, filter out archived ones, return them as a DataFrame or list.
    """
    raw_configs = _client.list_score_configs()
    active = [c for c in raw_configs if not c.get("isArchived")]
    # We'll return them as a list, or if you prefer a DataFrame:
    return active


def main():
    st.title("Langfuse Trace Reviewer + Dynamic Scores")

    # 1) Initialize client
    client = get_langfuse_client()

    # 2) Load trace data & score configs
    df = load_data(client)
    score_configs = load_active_score_configs(client)

    if df.empty:
        st.warning("No traces found matching the filter!")
        return

    st.subheader("Select a Trace")
    trace_ids = df["ID"].tolist()
    selected_trace_id = st.selectbox("Trace ID", options=trace_ids)
    row = df[df["ID"] == selected_trace_id].iloc[0]

    # Show relevant trace info
    st.write(f"**Timestamp:** {row['Timestamp']}")
    st.write(f"**Name / Tags:** {row['Name']} / {row['Tags']}")

    with st.expander("Conversation & Model Output", expanded=True):
        st.text_area("User Question", row["User Question"], height=150, disabled=True)
        st.text_area("Conversation", row["Conversation History"], height=150, disabled=True)
        st.text_area("Retrieved Context", row["Retrieved Context"], height=150, disabled=True)
        st.text_area("Model Thoughts", row["Model Thoughts"], height=150, disabled=True)
        st.text_area("Model Answer", row["Answer"], height=150, disabled=True)

    # 3) Ideal answer in metadata
    st.subheader("Ideal Answer Given Inputs (Metadata)")
    edited_ideal_answer = st.text_area(
        "ideal_answer_given_inputs",
        value=row["ideal_answer_given_inputs"],
        height=150
    )

    # 4) Dynamically show a UI for each active score config
    #    We'll store the user inputs in a dict {score_name -> new_value}.
    st.subheader("Scores")
    new_score_values = {}

    for config in score_configs:
        score_name = config["name"]
        data_type = config["dataType"]  # "CATEGORICAL", "NUMERIC", "BOOLEAN"
        categories = config.get("categories", [])  # only relevant if data_type == "CATEGORICAL"

        # The old/current value in the DF (if any). Could be empty if none found.
        old_val = row.get(score_name, "")
        label_for_score = f"{score_name} ({data_type})"

        if data_type == "CATEGORICAL":
            if categories:
                # Let's use the category labels as the selectbox choices
                # We'll find the label that matches old_val if possible
                all_labels = [cat["label"] for cat in categories]
                # If old_val is one of those labels, we default to it. Otherwise, default to the first or None.
                default_index = 0
                if old_val in all_labels:
                    default_index = all_labels.index(old_val)

                selected_label = st.selectbox(
                    label_for_score,
                    options=all_labels,
                    index=default_index if default_index < len(all_labels) else 0
                )
                # We'll store the selected label as the new value
                new_score_values[score_name] = selected_label
            else:
                # No categories defined? We can just treat it like a text input
                new_val = st.text_input(label_for_score, value=str(old_val))
                new_score_values[score_name] = new_val

        elif data_type == "NUMERIC":
            # If there's a minValue or maxValue
            min_val = config.get("minValue", None)
            max_val = config.get("maxValue", None)
            # Convert old_val to float if possible
            try:
                old_float = float(old_val)
            except ValueError:
                old_float = 0.0

            new_float = st.number_input(
                label_for_score,
                min_value=min_val if isinstance(min_val, (int, float)) else None,
                max_value=max_val if isinstance(max_val, (int, float)) else None,
                value=old_float
            )
            new_score_values[score_name] = new_float

        elif data_type == "BOOLEAN":
            # interpret old_val as True/False
            old_bool = str
