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
    print("Entered function: get_langfuse_client")
    client = LangfuseClient(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )
    print("Exiting function: get_langfuse_client")
    return client


@st.cache_data
def load_data(_client: LangfuseClient) -> pd.DataFrame:
    """
    Fetch the DataFrame of traces from the LangfuseClient.
    The leading underscore in _client tells Streamlit not to hash it.
    """
    print("Entered function: load_data")
    df = _client.load_traces_as_dataframe()
    print("Exiting function: load_data")
    return df


@st.cache_data
def load_active_score_configs(_client: LangfuseClient) -> list:
    """
    Fetch all score configs, filter out archived ones, return them as a list.
    """
    print("Entered function: load_active_score_configs")
    raw_configs = _client.list_score_configs()
    active = [c for c in raw_configs if not c.get("isArchived")]
    print("Exiting function: load_active_score_configs")
    return active


def main():
    print("Entered function: main")
    st.title("Langfuse Trace Reviewer + Dynamic Scores")

    client = get_langfuse_client()
    df = load_data(client)
    score_configs = load_active_score_configs(client)

    if df.empty:
        st.warning("No traces found matching the filter!")
        print("Exiting function: main (no traces found)")
        return

    st.subheader("Select a Trace")
    trace_ids = df["ID"].tolist()
    selected_trace_id = st.selectbox("Trace ID", options=trace_ids)
    row = df[df["ID"] == selected_trace_id].iloc[0]

    st.write(f"**Timestamp:** {row['Timestamp']}")
    st.write(f"**Name / Tags:** {row['Name']} / {row['Tags']}")

    with st.expander("Conversation & Model Output", expanded=True):
        st.text_area("User Question", row["User Question"], height=150, disabled=True)
        st.text_area("Conversation", row["Conversation History"], height=150, disabled=True)
        st.text_area("Retrieved Context", row["Retrieved Context"], height=150, disabled=True)
        st.text_area("Model Thoughts", row["Model Thoughts"], height=150, disabled=True)
        st.text_area("Model Answer", row["Answer"], height=150, disabled=True)

    st.subheader("Ideal Answer Given Inputs (Metadata)")
    edited_ideal_answer = st.text_area(
        "ideal_answer_given_inputs",
        value=row["ideal_answer_given_inputs"],
        height=150
    )

    st.subheader("Scores")
    new_score_values = {}

    for config in score_configs:
        score_name = config["name"]
        data_type = config["dataType"]
        categories = config.get("categories", [])

        old_val = row.get(score_name, "")
        label_for_score = f"{score_name} ({data_type})"

        if data_type == "CATEGORICAL":
            if categories:
                all_labels = ["<None>"] + [cat["label"] for cat in categories]
                default_index = 0
                if old_val in all_labels:
                    default_index = all_labels.index(old_val)

                selected_label = st.selectbox(
                    label_for_score,
                    options=all_labels,
                    index=default_index if default_index < len(all_labels) else 0
                )
                if selected_label == "<None>":
                    new_score_values[score_name] = None
                else:
                    new_score_values[score_name] = selected_label
            else:
                new_val = st.text_input(label_for_score, value=str(old_val))
                if not new_val.strip():
                    new_score_values[score_name] = None
                else:
                    new_score_values[score_name] = new_val

        elif data_type == "NUMERIC":
            min_val = config.get("minValue", None)
            max_val = config.get("maxValue", None)
            if isinstance(min_val, int):
                min_val = float(min_val)
            if isinstance(max_val, int):
                max_val = float(max_val)

            try:
                old_float = float(old_val)
            except ValueError:
                old_float = 0.0

            new_float = st.number_input(
                label_for_score,
                min_value=min_val if min_val is not None else None,
                max_value=max_val if max_val is not None else None,
                value=old_float
            )
            new_score_values[score_name] = new_float

        elif data_type == "BOOLEAN":
            old_bool = str(old_val).lower() in ("true", "yes", "1")
            new_bool = st.checkbox(label_for_score, value=old_bool)
            new_score_values[score_name] = new_bool
        else:
            new_val = st.text_input(label_for_score, value=str(old_val))
            if not new_val.strip():
                new_score_values[score_name] = None
            else:
                new_score_values[score_name] = new_val

    if st.button("Save Changes"):
        print("Save Changes clicked")
        # 1) Update the trace's ideal answer
        client.update_trace_ideal_answer(selected_trace_id, edited_ideal_answer)

        # 2) For each active score config, only create/update if value is not None
        for config in score_configs:
            score_name = config["name"]
            data_type = config["dataType"]
            new_val = new_score_values.get(score_name, None)

            if new_val is not None:
                if data_type == "CATEGORICAL":
                    client.create_or_update_score(
                        trace_id=selected_trace_id,
                        name=score_name,
                        data_type=data_type,
                        string_value=str(new_val),
                        value=None
                    )
                elif data_type == "NUMERIC":
                    try:
                        float_val = float(new_val)
                    except:
                        float_val = 0.0
                    client.create_or_update_score(
                        trace_id=selected_trace_id,
                        name=score_name,
                        data_type=data_type,
                        value=float_val,
                        string_value=None
                    )
                elif data_type == "BOOLEAN":
                    bool_val = bool(new_val)
                    client.create_or_update_score(
                        trace_id=selected_trace_id,
                        name=score_name,
                        data_type=data_type,
                        value=bool_val,
                        string_value=None
                    )
                else:
                    client.create_or_update_score(
                        trace_id=selected_trace_id,
                        name=score_name,
                        data_type="CATEGORICAL",
                        string_value=str(new_val),
                        value=None
                    )

        updated_df = load_data(client)
        client.export_to_csv(updated_df, OUTPUT_FILE)
        st.success("Trace metadata & scores updated successfully!")
        print("Save Changes completed")

    print("Exiting function: main")


if __name__ == "__main__":
    print("Running streamlit_app.py as __main__")
    main()
