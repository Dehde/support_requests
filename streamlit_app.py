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
    return _client.load_traces_as_dataframe()


def main():
    st.title("Langfuse Trace Reviewer / Editor")

    client = get_langfuse_client()
    df = load_data(client)
    if df.empty:
        st.warning("No traces found!")
        return

    # Let user pick a trace
    st.subheader("Select a trace to edit:")
    trace_ids = df["ID"].tolist()
    selected_trace_id = st.selectbox("Trace ID", options=trace_ids)

    # Grab the row
    row = df[df["ID"] == selected_trace_id].iloc[0]

    # Show some info
    st.write(f"**Timestamp:** {row['Timestamp']}")
    st.write(f"**Name/Tags:** {row['Name']} / {row['Tags']}")

    # Show conversation & answer
    with st.expander("Conversation & Model Output", expanded=True):
        st.text_area("User Question", row["User Question"], height=150, disabled=True)
        st.text_area("Conversation", row["Conversation History"], height=150, disabled=True)
        st.text_area("Retrieved Context", row["Retrieved Context"], height=150, disabled=True)
        st.text_area("Model Thoughts", row["Model Thoughts"], height=150, disabled=True)
        st.text_area("Model Answer", row["Answer"], height=150, disabled=True)

    # 1) Edit 'ideal_answer_given_inputs'
    st.subheader("Ideal Answer (Given Inputs)")
    edited_ideal_answer = st.text_area(
        "This is the perfect answer given the context at the time.",
        value=row["ideal_answer_given_inputs"] or "",
        height=150
    )

    # 2) Let user edit or add scores
    st.subheader("Scores")
    # For example, let's assume you have "HumanAnswerCorrectness" and "HumanActionNeeded".
    # If your DF has them, show them; else default to empty.
    default_hac = row.get("HumanAnswerCorrectness", "")
    default_han = row.get("HumanActionNeeded", "")

    # Could be text inputs, numeric inputs, or selectboxes, up to you.
    # Example: a text input for correctness, a checkbox for action needed
    # (But let's keep them simple as text for demonstration.)
    new_hac = st.text_input("HumanAnswerCorrectness", value=str(default_hac))
    new_han = st.text_input("HumanActionNeeded", value=str(default_han))

    # Save button
    if st.button("Save Changes"):
        # 1) Update the trace's metadata with the new ideal_answer
        client.update_trace_ideal_answer(selected_trace_id, edited_ideal_answer)

        # 2) Upsert the scores (if user typed anything)
        if new_hac.strip():
            # We can decide data_type='CATEGORICAL' or 'NUMERIC' etc.
            # If the user typed a string, we might do stringValue= that string.
            client.create_or_update_score(
                trace_id=selected_trace_id,
                name="HumanAnswerCorrectness",
                data_type="CATEGORICAL",
                string_value=new_hac.strip()
            )

        if new_han.strip():
            client.create_or_update_score(
                trace_id=selected_trace_id,
                name="HumanActionNeeded",
                data_type="CATEGORICAL",
                string_value=new_han.strip()
            )

        # Optionally re-fetch data and store locally
        updated_df = load_data(client)
        client.export_to_csv(updated_df, OUTPUT_FILE)

        st.success("Metadata and Scores updated successfully!")


if __name__ == "__main__":
    main()
