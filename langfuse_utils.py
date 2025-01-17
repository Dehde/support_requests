# langfuse_utils.py

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any
from collections import defaultdict
from langfuse import Langfuse
from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

START_DATE = "2024-12-17"
TARGET_TAG = "app_id=d6bfd7f4-39a0-4824-8720-a8b79d32f586"
OUTPUT_FILE = "traces_processed.csv"


class LangfuseClient:
    """
    Encapsulates logic for:
      - Initializing the Langfuse() instance
      - Fetching & filtering traces
      - Managing 'ideal_answer_given_inputs' in trace.metadata
      - Listing/creating/updating scores
      - Building a DataFrame for the UI
      - Listing score configs
    """

    def __init__(self, public_key: str, secret_key: str, host: str):
        print("Entered function: LangfuseClient.__init__")
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self._client = Langfuse(public_key=self.public_key, secret_key=self.secret_key, host=self.host)

    def fetch_filtered_traces(self) -> List[TraceWithDetails]:
        print("Entered function: fetch_filtered_traces")
        start_dt = pd.to_datetime(START_DATE, utc=True)
        end_dt = pd.Timestamp.now(tz='UTC')
        all_traces = []
        limit = 100
        page = 1
        
        while True:
            response = self._client.fetch_traces(tags=[TARGET_TAG], from_timestamp=start_dt, to_timestamp=end_dt, limit=limit, page=page)
            fresh_traces = response.data
            all_traces.extend(fresh_traces)
            print(f"Fetched {len(fresh_traces)} trace(s) for page={page}")
            
            if len(response.data) < limit:
                break
            page += 1
            
        print(f"Exiting function: fetch_filtered_traces. Total: {len(all_traces)} trace(s).")
        return all_traces

    def load_traces_as_dataframe(self) -> pd.DataFrame:
        print("Entered function: load_traces_as_dataframe")
        traces = self.fetch_filtered_traces()
        scores = self.request_scores()
        scores_map = self.build_trace_scores_map(scores)
        rows = self.create_data_rows(traces, scores_map)
        df = pd.DataFrame(rows)

        # Deduplicate based on 'User Question', keeping the first occurrence
        df = df.drop_duplicates(subset=['User Question'], keep='first')

        preferred_cols = [
            "ID", "Timestamp", "User Question", "Conversation History",
            "Retrieved Context", "Model Thoughts", "Answer", "Expected Answer",
            "ideal_answer_given_inputs",
            "Name", "Tags"
        ]
        existing_cols = [c for c in preferred_cols if c in df.columns]
        df = df.reindex(columns=existing_cols + [c for c in df.columns if c not in existing_cols])
        print("Exiting function: load_traces_as_dataframe")
        return df

    def update_trace_ideal_answer(self, trace_id: str, ideal_answer: str, ideal_answer_given_inputs: str, comment: str) -> None:
        """
        Upsert the trace with metadata including ideal answers and comment.
        """
        print("Entered function: update_trace_ideal_answer")
        trace_obj = self._client.get_trace(trace_id)
        old_metadata = trace_obj.metadata or {}
        old_metadata["ideal_answer"] = ideal_answer
        old_metadata["ideal_answer_given_inputs"] = ideal_answer_given_inputs
        old_metadata["comment"] = comment
        self._client.trace(id=trace_obj.id, metadata=old_metadata)
        print("Exiting function: update_trace_ideal_answer")

    def request_scores(self) -> List[Dict[str, Any]]:
        print("Entered function: request_scores")
        resp = requests.get(
            f"{self.host}/api/public/scores",
            auth=HTTPBasicAuth(self.public_key, self.secret_key),
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        print("Exiting function: request_scores")
        return data

    def build_trace_scores_map(self, scores: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        print("Entered function: build_trace_scores_map")
        trace_scores_map = defaultdict(list)
        for sc in scores:
            tid = sc.get("traceId")
            if tid:
                trace_scores_map[tid].append(sc)
        print("Exiting function: build_trace_scores_map")
        return trace_scores_map

    def create_or_update_score(
            self,
            trace_id: str,
            name: str,
            data_type: str,
            value=None,
            string_value=None,
            score_id=None
    ) -> None:
        """
        Use the official API approach:
          - Always pass "value".
          - If data_type=NUMERIC or BOOLEAN, 'value' is numeric (float or int).
          - If data_type=CATEGORICAL, 'value' is string.
          - 'comment' is optional, if you'd like to store extra info.
          - 'id' can be passed if you want to upsert by an idempotency key.

        For BOOLEANS, per the doc, value is numeric: 1 -> True, 0 -> False,
        plus the server sets stringValue to "True"/"False".
        """
        print("Entered function: create_or_update_score")
        score_kwargs = {
            "trace_id": trace_id,
            "name": name,
            "dataType": data_type,
        }

        # If you want to do upserts by ID (idempotency key):
        if score_id:
            score_kwargs["id"] = score_id

        # Decide how to pass 'value' based on data_type
        # We'll interpret new_val differently for numeric/boolean vs. categorical

        if data_type == "NUMERIC":
            # value must be numeric
            if value is None:
                # default to 0
                score_kwargs["value"] = 0.0
            else:
                score_kwargs["value"] = float(value)
            # optionally, you might have some 'string_value' to store as well
            # but in the doc, 'comment' is separate from 'value'
            if string_value:
                score_kwargs["comment"] = str(string_value)

        elif data_type == "BOOLEAN":
            # value must be numeric: 1 => True, 0 => False
            bool_val = bool(value) if value is not None else False
            score_kwargs["value"] = 1 if bool_val else 0
            # The server automatically sets stringValue to "True" or "False"
            if string_value:
                score_kwargs["comment"] = str(string_value)

        elif data_type == "CATEGORICAL":
            # value must be a string
            # if you pass numeric, you'd get an error
            # We interpret 'value' or 'string_value' as the final string
            # For safety, let's always rely on 'string_value'
            if string_value is None and value is not None:
                # fallback => cast numeric to str
                score_kwargs["value"] = str(value)
            else:
                # typical path
                str_val = str(string_value) if string_value is not None else ""
                score_kwargs["value"] = str_val
            # 'comment' is optional but can be used for additional notes
            # e.g. if you want to store "some extra detail"
            # score_kwargs["comment"] = "some optional extra detail"

        else:
            # fallback or unknown type
            # we could default to CATEGORICAL logic, or raise an error
            # for safety, let's treat it as text
            print(f"Warning: unknown data_type={data_type}, defaulting to string.")
            score_kwargs["dataType"] = "CATEGORICAL"
            str_val = str(string_value) if string_value is not None else str(value) if value else ""
            score_kwargs["value"] = str_val

        print(f"Sending score with: {score_kwargs}")
        self._client.score(**score_kwargs)
        print("Exiting function: create_or_update_score")

    def list_score_configs(self) -> List[Dict[str, Any]]:
        print("Entered function: list_score_configs")
        url = f"{self.host}/api/public/score-configs"
        resp = requests.get(
            url,
            auth=HTTPBasicAuth(self.public_key, self.secret_key),
            headers={"Content-Type": "application/json"},
            params={"page": 1, "limit": 100}
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        print("Exiting function: list_score_configs")
        return data

    # -------------------------------------------------------------------------
    # 3) BUILD DATAFRAME ROWS
    # -------------------------------------------------------------------------
    def create_data_rows(
            self,
            traces: List[TraceWithDetails],
            trace_scores_map: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        print("Entered function: create_data_rows")
        rows = []
        for trace in traces:
            ts_str = trace.timestamp.strftime("%Y-%m-%d %H:%M:%S")  # Changed here
            user_q = trace.metadata.get("user_question", "")
            model_thoughts = trace.metadata.get("model_thoughts", "")
            conv_entries = trace.metadata.get("conversation_history") or []
            conversation_history = "\n".join(
                f"{e.get('role', 'unknown')}: {e.get('content', '')}"
                for e in conv_entries
            )
            retrieved_contexts = trace.metadata.get("retrieved_contexts", [])
            retrieved_contexts_sorted = sorted(
                retrieved_contexts, key=lambda c: c.get("cosine_distance", 9999)
            )[:5]
            ctx_text = ""
            for i, ctx in enumerate(retrieved_contexts_sorted, start=1):
                link = ctx.get("link", "No link")
                dist = ctx.get("cosine_distance", "N/A")
                ctx_text += f"[Context {i}]\nLink: {link}\nCosine Distance: {dist}\n\n"

            original_answer = trace.output if trace.output else ""
            ideal_answer = trace.metadata.get("ideal_answer", "")
            ideal_answer_given_inputs = trace.metadata.get("ideal_answer_given_inputs", "")

            row = {
                "ID": trace.id,
                "Timestamp": ts_str,
                "User Question": user_q,
                "Conversation History": conversation_history,
                "Retrieved Context": ctx_text.strip(),
                "Model Thoughts": model_thoughts,
                "Answer": original_answer,
                "Expected Answer": original_answer,
                "ideal_answer": ideal_answer,
                "ideal_answer_given_inputs": ideal_answer_given_inputs,
                "comment": trace.metadata.get("comment", ""),
                "Name": trace.name,
                "Tags": ", ".join(trace.tags) if trace.tags else "",
            }

            # Attach any existing scores
            scores_for_trace = trace_scores_map.get(trace.id, [])
            for sc in scores_for_trace:
                name = sc.get("name", "")
                val = sc.get("value", None)
                com = sc.get("comment", None)
                if com is not None and com != "":
                    row[name] = com
                elif val is not None:
                    row[name] = str(val)
                else:
                    row[name] = ""

            rows.append(row)

        print("Exiting function: create_data_rows")
        return rows

    # -------------------------------------------------------------------------
    # CSV Export (optional)
    # -------------------------------------------------------------------------
    def export_to_csv(self, df: pd.DataFrame, file_name: str = OUTPUT_FILE) -> None:
        print("Entered function: export_to_csv")
        df.to_csv(file_name, index=False)
        print("Exiting function: export_to_csv")

    def update_trace_in_df(self, df: pd.DataFrame, trace_id: str) -> pd.DataFrame:
        """
        Updates a single trace in the DataFrame with fresh data from Langfuse.
        Returns the updated DataFrame.
        """
        print(f"Entered function: update_trace_in_df for trace_id={trace_id}")
        trace = self._client.fetch_trace(trace_id).data
        scores = self.request_scores()
        scores_map = self.build_trace_scores_map(scores)
        updated_row = self.create_data_rows([trace,], scores_map)[0]
        updated_row = pd.Series(updated_row).reindex(df.columns, fill_value=np.nan)
        old_id = df.loc[df['ID'] == trace_id].index[0]
        df.loc[old_id, df.columns] = pd.Series(updated_row).reindex(df.columns, fill_value=np.nan).values
        print("Exiting function: update_trace_in_df")
        return df


if __name__ == "__main__":
    import sys
    import streamlit as st

    LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
    LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
    LANGFUSE_HOST = st.secrets["LANGFUSE_HOST"]

    client = LangfuseClient(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
    try:
        print("Loading traces as DataFrame...")
        df = client.load_traces_as_dataframe()
        print(f"Loaded {len(df)} traces.")

        # Test updating a single trace - let's use the first one
        if not df.empty:
            test_trace_id = df.iloc[0]['ID']
            print(f"\nTesting single trace update for ID: {test_trace_id}")
            updated_df = client.update_trace_in_df(df, test_trace_id)
            print("Successfully updated trace in DataFrame")

        print(f"\nExporting DataFrame to {OUTPUT_FILE}...")
        client.export_to_csv(df)
        print(f"Successfully exported traces to {OUTPUT_FILE}.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)