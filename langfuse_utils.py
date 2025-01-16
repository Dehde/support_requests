# langfuse_utils.py

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any
from collections import defaultdict
from langfuse import Langfuse
from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails

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
        self._client = None
        print("Exiting function: LangfuseClient.__init__")

    def _init_client(self) -> Langfuse:
        print("Entered function: _init_client")
        if not self._client:
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
        print("Exiting function: _init_client")
        return self._client

    # -------------------------------------------------------------------------
    # 1) TRACES & METADATA
    # -------------------------------------------------------------------------
    def fetch_filtered_traces(self) -> List[TraceWithDetails]:
        print("Entered function: fetch_filtered_traces")
        client = self._init_client()
        # Convert your START_DATE to a UTC-aware datetime if it isn't already
        start_dt = pd.to_datetime(START_DATE, utc=True)
        all_traces = []
        limit = 50  # The API default is typically 50, you can adjust as needed
        page = 1
        # Keep fetching traces until a page returns fewer than 'limit' results
        while True:
            response = client.fetch_traces(
                tags=[TARGET_TAG],
                from_timestamp=start_dt,
                limit=limit,
                page=page
            )
            all_traces.extend(response.data)
            print(f"Fetched {len(response.data)} trace(s) for page={page}")
            # If the response has fewer than 'limit' items, it's the last page
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

    def update_trace_ideal_answer(self, trace_id: str, ideal_answer: str) -> None:
        """
        Upsert the trace with updated metadata['ideal_answer_given_inputs']
        using langfuse.trace().
        """
        print("Entered function: update_trace_ideal_answer")
        lf_client = self._init_client()

        # 1) Fetch the existing trace to avoid overwriting other metadata.
        trace_obj = lf_client.get_trace(trace_id)
        old_metadata = trace_obj.metadata or {}

        # 2) Merge your new/edited field
        old_metadata["ideal_answer_given_inputs"] = ideal_answer

        # 3) Upsert using langfuse.trace(...)
        lf_client.trace(
            id=trace_obj.id,
            metadata=old_metadata
        )
        print("Exiting function: update_trace_ideal_answer")

    # -------------------------------------------------------------------------
    # 2) SCORES & SCORE CONFIG
    # -------------------------------------------------------------------------
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
        lf_client = self._init_client()

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
        lf_client.score(**score_kwargs)
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
            # Include seconds in your timestamp
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
            ideal_answer = trace.metadata.get("ideal_answer_given_inputs", "")

            row = {
                "ID": trace.id,
                "Timestamp": ts_str,
                "User Question": user_q,
                "Conversation History": conversation_history,
                "Retrieved Context": ctx_text.strip(),
                "Model Thoughts": model_thoughts,
                "Answer": original_answer,
                "Expected Answer": original_answer,
                "ideal_answer_given_inputs": ideal_answer,
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

        print(f"Exporting DataFrame to {OUTPUT_FILE}...")
        client.export_to_csv(df)
        print(f"Successfully exported traces to {OUTPUT_FILE}.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)