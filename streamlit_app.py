# streamlit_app.py

import streamlit as st
import pandas as pd

# Import our refactored code
from langfuse_utils import LangfuseClient, OUTPUT_FILE
from trace_model import LangfuseTrace

# We rely on st.secrets for these
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST = st.secrets["LANGFUSE_HOST"]

# -------------
# UI Helpers
# -------------
def display_trace_details_ui(df: pd.DataFrame, selected_trace_id: str) -> str:
    """
    Given the DataFrame and a chosen trace_id, display the trace
    details (user Q, conversation, context, thoughts, answer),
    returning the (possibly updated) 'Expected Answer' from the user.
    """
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
    return expected_answer


def main():
    st.title("Langfuse Trace Reviewer (Refactored)")

    # 1) Initialize our client
    @st.cache_resource
    def get_client() -> LangfuseClient:
        return LangfuseClient(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )

    langfuse_client = get_client()

    # 2) Load the trace data as a DataFrame
    #    (We rely on the client method, which calls the Langfuse API.)
    @st.cache_data
    def load_data() -> pd.DataFrame:
        return langfuse_client.load_traces_as_dataframe()

    df = load_data()
    if df.empty:
        st.warning("No traces found!")
        return

    # 3) Dataset selection UI
    st.subheader("Pick or create a dataset:")
    datasets = langfuse_client.list_datasets()
    dataset_names = [ds["name"] for ds in datasets] if datasets else []

    col_left, col_right = st.columns([2, 2])
    with col_left:
        selected_dataset = st.selectbox(
            "Existing Datasets",
            options=["<None>"] + dataset_names,
            index=0
        )
    with col_right:
        new_dataset_name = st.text_input("Or create a new dataset", "")

    if new_dataset_name.strip():
        chosen_dataset = new_dataset_name.strip()
    elif selected_dataset != "<None>":
        chosen_dataset = selected_dataset
    else:
        chosen_dataset = None

    # 4) Trace selection
    st.subheader("Select a trace to review:")
    trace_ids = df["ID"].tolist()
    selected_trace_id = st.selectbox("Trace", options=trace_ids, format_func=lambda x: f"Trace: {x}")

    # 5) Display the selected trace details in read-only text areas, except 'Expected Answer'
    expected_answer = display_trace_details_ui(df, selected_trace_id)

    # 6) Save button
    if st.button("Save Dataset Item"):
        if not chosen_dataset:
            st.error("Please pick a dataset or enter a new dataset name.")
            return

        # Update local DataFrame with new expected answer
        df.loc[df["ID"] == selected_trace_id, "Expected Answer"] = expected_answer
        # Optionally export locally
        langfuse_client.export_to_csv(df, OUTPUT_FILE)

        # Ensure dataset exists or create it
        ds_obj = langfuse_client.get_or_create_dataset(chosen_dataset)

        # Prepare input data for the item
        trace_row = df[df["ID"] == selected_trace_id].iloc[0]
        input_data = {
            "user_question": trace_row["User Question"],
            "conversation_history": trace_row["Conversation History"],
            "context": trace_row["Retrieved Context"]
        }

        # Upsert item into the dataset
        langfuse_client.create_or_update_dataset_item(
            dataset_name=ds_obj["name"],
            item_id=selected_trace_id,
            input_data=input_data,
            expected_answer=expected_answer
        )

        st.success(
            f"Trace '{selected_trace_id}' saved in dataset '{ds_obj['name']}' with new Expected Answer!"
        )


if __name__ == "__main__":
    main()
