import streamlit as st
import pandas as pd
from datetime import datetime

from langfuse_utils import LangfuseClient, OUTPUT_FILE

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
def get_current_date_key() -> str:
    """Returns today's date as a string - will be different each day"""
    return datetime.now().strftime("%Y-%m-%d")


def load_data_for_date(_client: LangfuseClient) -> pd.DataFrame:
    """Loads data from API - cached separately for each date"""
    return _client.load_traces_as_dataframe()


@st.cache_data
def load_active_score_configs(_client: LangfuseClient) -> list:
    print("Entered function: load_active_score_configs")
    raw_configs = _client.list_score_configs()
    active = [c for c in raw_configs if not c.get("isArchived")]
    print("Exiting function: load_active_score_configs")
    return active


def main():
    print("Entered function: main")
    st.title("Langfuse Trace Reviewer + Dynamic Scores")

    client = get_langfuse_client()
    today_key = get_current_date_key()
    
    # Initialize or update session state based on date
    if "current_df" not in st.session_state or "last_load_date" not in st.session_state or st.session_state.last_load_date != today_key:
        st.session_state.current_df = load_data_for_date(client)
        st.session_state.last_load_date = today_key
    
    df = st.session_state.current_df
    score_configs = load_active_score_configs(client)

    # 3) Add a checkbox to filter unreviewed traces
    st.sidebar.header("Filters")
    show_only_unreviewed = st.sidebar.checkbox(
        "Show only traces needing review",
        value=True,
        help="Check this to display only traces that haven't been reviewed yet."
    )
    df_not_reviewed = df[df["ideal_answer"].isnull() | (df["ideal_answer"] == "")]
    st.sidebar.write(f"**Traces needing review:** {df_not_reviewed.shape[0]}")
    st.sidebar.write(f"**Total traces:** {df.shape[0]}")

    if show_only_unreviewed:
        filtered_df = df_not_reviewed
        
    else:
        filtered_df = df

    if filtered_df.empty:
        st.warning("No traces found matching the current filter!")
        print("Exiting function: main (no traces found after filtering)")
        return

    # 4) Sort by timestamp DESC and select by index
    filtered_df = filtered_df.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)

    st.subheader("Select a Trace")

    # Convert Timestamp column to datetime if it's not already
    filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])
    
    # Get unique months and format them
    unique_months = filtered_df['Timestamp'].dt.to_period('M').unique()
    month_options = sorted(unique_months, reverse=True)  # Most recent first
    
    selected_month = st.selectbox(
        "Select Month",
        options=month_options,
        format_func=lambda x: x.strftime("%b '%y")  # Format like "Jan '24"
    )

    # Filter traces for selected month
    month_traces = filtered_df[filtered_df['Timestamp'].dt.to_period('M') == selected_month]
    month_traces = month_traces.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
    
    # Add monthly statistics to sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Monthly Statistics")
    monthly_counts = filtered_df.groupby(filtered_df['Timestamp'].dt.to_period('M')).size()
    for month in sorted(monthly_counts.index, reverse=True):
        count = monthly_counts[month]
        st.sidebar.write(f"**{month.strftime('%b %Y')}:** {count} traces")

    # Create timestamp-question selection for the selected month
    def format_trace_option(index):
        row = month_traces.loc[index]
        timestamp = row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        question_preview = row["User Question"][:50] + "..." if len(row["User Question"]) > 50 else row["User Question"]
        return f"{timestamp} - {question_preview}"

    selected_index = st.selectbox(
        "Select Trace",
        options=month_traces.index,
        format_func=format_trace_option
    )

    selected_trace_id = month_traces.loc[selected_index]["ID"]
    print(selected_trace_id)
    
    # Add trace ID display
    st.write(f"**Trace ID:** {selected_trace_id}")
    
    st.session_state.current_df = client.update_trace_in_df(st.session_state.current_df, selected_trace_id)
    row = st.session_state.current_df[st.session_state.current_df['ID'] == selected_trace_id].iloc[0]

    # 5) Display trace information
    st.write(f"**Timestamp:** {row['Timestamp']}")
    st.write(f"**Name / Tags:** {row['Name']} / {row['Tags']}")

    with st.expander("Conversation & Model Output", expanded=True):
        # For read-only fields, we usually don't need dynamic keys (they won't preserve user edits).
        # But it's fine either way if you want them to update reliably.
        st.text_area("User Question", row["User Question"], height=150, disabled=True)
        st.text_area("Conversation", row["Conversation History"], height=150, disabled=True)
        st.text_area("Retrieved Context", row["Retrieved Context"], height=150, disabled=True)
        st.text_area("Model Thoughts", row["Model Thoughts"], height=150, disabled=True)
        st.text_area("Model Answer", row["Answer"], height=150, disabled=True)

    # 6) Edit Ideal Answers and Comment - give them dynamic keys
    st.subheader("Ideal Answers and Comments (Metadata)")
    
    edited_ideal_answer = st.text_area(
        "ideal_answer (The best possible answer regardless of context)",
        value=row["ideal_answer"],
        height=150,
        key=f"ideal_answer_{selected_trace_id}"
    )
    
    edited_ideal_answer_given_inputs = st.text_area(
        "ideal_answer_given_inputs (The best possible answer given the retrieved context)",
        value=row["ideal_answer_given_inputs"],
        height=150,
        key=f"ideal_answer_given_inputs_{selected_trace_id}"
    )

    edited_comment = st.text_area(
        "Comment (Any additional notes or observations)",
        value=row.get("comment", ""),
        height=150,
        key=f"comment_{selected_trace_id}"
    )

    # 7) Dynamic Score Inputs
    st.subheader("Scores")
    new_score_values = {}

    for config in score_configs:
        score_name = config["name"]
        data_type = config["dataType"]
        categories = config.get("categories", [])

        old_val = row.get(score_name, "")  # existing score value from DF
        label_for_score = f"{score_name} ({data_type})"

        # We'll include the trace ID in the key so each trace gets its own widget state
        dynamic_key = f"{score_name}_{selected_trace_id}"

        if data_type == "CATEGORICAL":
            if categories:
                all_labels = ["<None>"] + [cat["label"] for cat in categories]
                default_index = 0
                if old_val in all_labels:
                    default_index = all_labels.index(old_val)

                selected_label = st.selectbox(
                    label_for_score,
                    options=all_labels,
                    index=default_index if default_index < len(all_labels) else 0,
                    key=dynamic_key
                )
                if selected_label == "<None>":
                    new_score_values[score_name] = None
                else:
                    new_score_values[score_name] = selected_label
            else:
                new_val = st.text_input(label_for_score, value=str(old_val), key=dynamic_key)
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
                value=old_float,
                key=dynamic_key
            )
            new_score_values[score_name] = new_float

        elif data_type == "BOOLEAN":
            old_bool = str(old_val).lower() in ("true", "yes", "1")
            new_bool = st.checkbox(label_for_score, value=old_bool, key=dynamic_key)
            new_score_values[score_name] = new_bool
        else:
            # Fallback / unknown data type => treat as text
            new_val = st.text_input(label_for_score, value=str(old_val), key=dynamic_key)
            if not new_val.strip():
                new_score_values[score_name] = None
            else:
                new_score_values[score_name] = new_val

    # 8) Save Changes Button
    if st.button("Save Changes"):
        print("Save Changes clicked")
        # 1) Update the trace's ideal answers and comment
        client.update_trace_ideal_answer(
            selected_trace_id, 
            edited_ideal_answer,
            edited_ideal_answer_given_inputs,
            edited_comment
        )

        # 2) For each active score config, only create/update if value is not None
        for config in score_configs:
            score_name = config["name"]
            data_type = config["dataType"]
            new_val = new_score_values.get(score_name, None)

            if new_val is not None:
                # Prepare the value based on data type
                if data_type == "NUMERIC":
                    try:
                        value = float(new_val)
                    except:
                        value = 0.0
                elif data_type == "BOOLEAN":
                    value = bool(new_val)
                else:  # CATEGORICAL or fallback
                    value = str(new_val)
                    data_type = "CATEGORICAL"  # Force unknown types to CATEGORICAL

                kwargs = {
                    "trace_id": selected_trace_id,
                    "name": score_name,
                    "data_type": data_type,
                }
                kwargs["string_value" if data_type == "CATEGORICAL" else "value"] = value
                client.create_or_update_score(**kwargs)

        # Update session state DataFrame
        st.session_state.current_df.loc[df["ID"] == selected_trace_id, "ideal_answer"] = edited_ideal_answer
        st.session_state.current_df.loc[df["ID"] == selected_trace_id, "ideal_answer_given_inputs"] = edited_ideal_answer_given_inputs
        st.session_state.current_df.loc[df["ID"] == selected_trace_id, "comment"] = edited_comment
        
        # Update scores in the session state DataFrame
        for config in score_configs:
            score_name = config["name"]
            new_val = new_score_values.get(score_name, None)
            if new_val is not None:
                st.session_state.current_df.loc[df["ID"] == selected_trace_id, score_name] = new_val

        client.export_to_csv(st.session_state.current_df, OUTPUT_FILE)
        st.success("Trace data and scores updated successfully!")

    print("Exiting function: main")


if __name__ == "__main__":
    print("Running streamlit_app.py as __main__")
    main()
