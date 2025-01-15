# langfuse_utils.py

import pandas as pd
import requests
import streamlit
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any
from collections import defaultdict
from langfuse import Langfuse
from trace_model import LangfuseTrace

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
      - Building a DataFrame for UI display
    """

    def __init__(self, public_key: str, secret_key: str, host: str):
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self._client = None

    def _init_client(self) -> Langfuse:
        """ Lazily initialize the actual Langfuse client. """
        if not self._client:
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
        return self._client

    # -------------------------------------------------------------------------
    # Traces / Metadata
    # -------------------------------------------------------------------------
    def fetch_filtered_traces(self) -> List[LangfuseTrace]:
        """
        Fetch all traces with TARGET_TAG since START_DATE, wrap them in a LangfuseTrace dataclass.
        """
        client = self._init_client()
        start_dt = pd.to_datetime(START_DATE, utc=True)
        response = client.fetch_traces(tags=[TARGET_TAG])
        all_traces = response.data

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

    def load_traces_as_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing relevant trace info (including existing 'ideal_answer_given_inputs' if found).
        """
        traces = self.fetch_filtered_traces()
        scores = self.request_scores()
        scores_map = self.build_trace_scores_map(scores)
        rows = self.create_data_rows(traces, scores_map)
        df = pd.DataFrame(rows)

        preferred_cols = [
            "ID", "Timestamp", "User Question", "Conversation History",
            "Retrieved Context", "Model Thoughts", "Answer", "Expected Answer",
            "ideal_answer_given_inputs",  # We'll stick the trace metadata here if found
            "Name", "Tags", "user_feedback", "HumanAnswerCorrectness", "HumanActionNeeded"
        ]
        existing_cols = [c for c in preferred_cols if c in df.columns]
        df = df.reindex(columns=existing_cols + [c for c in df.columns if c not in existing_cols])
        return df

    def update_trace_ideal_answer(self, trace_id: str, ideal_answer: str) -> None:
        """
        Updates the trace with a key 'ideal_answer_given_inputs' in metadata.
        If you want to store a different key name, rename here.
        """
        client = self._init_client()

        # 1) Fetch the existing trace data (to not overwrite other metadata)
        trace = client.traces.get(trace_id)
        updated_metadata = trace.metadata or {}
        updated_metadata["ideal_answer_given_inputs"] = ideal_answer

        # 2) Update the trace. The client may have a direct method or we create an "observation".
        #    If the client has no direct 'update_trace()' method, we might do something like:
        trace_update_input = {
            "metadata": updated_metadata
        }
        client.traces.update(trace_id, trace_update_input)

    # -------------------------------------------------------------------------
    # Scores
    # -------------------------------------------------------------------------
    def request_scores(self) -> List[Dict[str, Any]]:
        """
        Return all existing scores from the public API.
        Each score has { "id", "traceId", "name", "dataType", "value" or "stringValue", ... }
        """
        resp = requests.get(
            f"{self.host}/api/public/scores",
            auth=HTTPBasicAuth(self.public_key, self.secret_key),
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    def build_trace_scores_map(self, scores: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build a map: trace_id -> list of these score dicts
        """
        trace_scores_map = defaultdict(list)
        for sc in scores:
            tid = sc.get("traceId")
            if tid:
                trace_scores_map[tid].append(sc)
        return trace_scores_map

    def create_or_update_score(
        self,
        trace_id: str,
        name: str,
        data_type: str,
        value=None,
        string_value=None
    ) -> None:
        """
        Creates or updates a score on the given trace.
        The same 'name' + 'trace_id' typically upserts in Langfuse.
        """
        client = self._init_client()
        client.create_score(
            trace_id=trace_id,
            name=name,
            data_type=data_type,
            value=value,           # numeric or boolean
            string_value=string_value
        )

    # -------------------------------------------------------------------------
    # Utility: build DataFrame rows
    # -------------------------------------------------------------------------
    def create_data_rows(
        self,
        traces: List[LangfuseTrace],
        trace_scores_map: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Convert the traces + scores into row dicts for a DataFrame.
        We'll also look if there's 'ideal_answer_given_inputs' in metadata.
        """
        allowed_scores = {"user_feedback", "HumanAnswerCorrectness", "HumanActionNeeded"}
        rows = []
        for trace in traces:
            ts_str = trace.timestamp.strftime("%Y-%m-%d %H:%M")
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
                "Expected Answer": original_answer,  # for display if you want
                "ideal_answer_given_inputs": ideal_answer,  # from metadata
                "Name": trace.name,
                "Tags": ", ".join(trace.tags) if trace.tags else "",
            }

            # Attach relevant scores
            scores_for_trace = trace_scores_map.get(trace.id, [])
            for sc in scores_for_trace:
                name = sc.get("name", "")
                dt = sc.get("dataType", "")
                if name in allowed_scores:
                    if dt == "CATEGORICAL":
                        row[name] = sc.get("stringValue", "")
                    else:
                        # Could be numeric or boolean
                        row[name] = sc.get("value", "")

            rows.append(row)
        return rows

    def export_to_csv(self, df: pd.DataFrame, file_name: str = OUTPUT_FILE) -> None:
        df.to_csv(file_name, index=False)
