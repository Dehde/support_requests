# langfuse_utils.py

import pandas as pd
import requests
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
    def fetch_filtered_traces(self) -> List[LangfuseTrace]:
        print("Entered function: fetch_filtered_traces")
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
        print("Exiting function: fetch_filtered_traces")
        return filtered

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
        string_value=None
    ) -> None:
        """
        Replaces client.create_score(...) with the documented langfuse.score(...) approach.

        We'll store numeric/boolean in 'value',
        and textual/categorical stuff in 'comment'.
        """
        print("Entered function: create_or_update_score")

        lf_client = self._init_client()

        # Decide how to pass the data:
        # - If numeric or boolean => put that in value
        # - If textual => store it in comment
        # (If your doc or your method differs, adjust accordingly.)
        score_kwargs = {
            "trace_id": trace_id,
            "name": name
        }

        if data_type in ("NUMERIC", "BOOLEAN"):
            # numeric or boolean => interpret 'value' as float/bool,
            # and ignore string_value if present
            if value is not None:
                score_kwargs["value"] = value
            else:
                # fallback if no numeric => set 0 or something
                score_kwargs["value"] = 0
        else:
            # e.g. CATEGORICAL or fallback
            # store the textual representation in 'comment'
            # ignoring numeric 'value'
            if string_value is not None:
                score_kwargs["comment"] = string_value
            else:
                score_kwargs["comment"] = ""

        # Now call `lf_client.score(...)`
        # e.g.:
        # lf_client.score(trace_id=..., name=..., value=..., comment=...)
        lf_client.score(**score_kwargs)

        print(f"Called lf_client.score with kwargs={score_kwargs}")
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
        traces: List[LangfuseTrace],
        trace_scores_map: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        print("Entered function: create_data_rows")
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
                "Expected Answer": original_answer,
                "ideal_answer_given_inputs": ideal_answer,
                "Name": trace.name,
                "Tags": ", ".join(trace.tags) if trace.tags else "",
            }

            # Attach any existing scores
            scores_for_trace = trace_scores_map.get(trace.id, [])
            for sc in scores_for_trace:
                name = sc.get("name", "")
                # sc might have "value", "comment", etc.
                # We'll do a guess: If 'comment' is present, let's store that.
                # Otherwise store 'value'.
                val = sc.get("value", None)
                com = sc.get("comment", None)

                # We'll decide how to display it. If there's a comment, use it, else string of val
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
