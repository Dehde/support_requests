# langfuse_utils.py

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any
from collections import defaultdict
from langfuse import Langfuse
from trace_model import LangfuseTrace

# Constants
START_DATE = "2024-12-17"
TARGET_TAG = "app_id=d6bfd7f4-39a0-4824-8720-a8b79d32f586"
OUTPUT_FILE = "traces_processed.csv"


class LangfuseClient:
    """
    Wraps logic related to:
      - Initializing and caching a Langfuse() instance
      - Fetching & filtering traces
      - Requesting scores
      - Building a DataFrame from trace data
      - Listing/creating datasets
      - Creating/updating dataset items
    """

    def __init__(self, public_key: str, secret_key: str, host: str):
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self._client = None  # Will lazy-init on first usage

    def _init_client(self) -> Langfuse:
        """
        Lazily initialize the actual langfuse.Langfuse client.
        """
        if not self._client:
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
        return self._client

    def fetch_filtered_traces(self) -> List[LangfuseTrace]:
        """
        Fetch all traces with TARGET_TAG since START_DATE, wrap them in our LangfuseTrace dataclass.
        """
        client = self._init_client()
        start_dt = pd.to_datetime(START_DATE, utc=True)

        # Retrieve from Langfuse
        traces_response = client.fetch_traces(tags=[TARGET_TAG])
        all_traces = traces_response.data

        filtered = []
        for t in all_traces:
            trace_dt_utc = pd.to_datetime(t.timestamp, utc=True)
            if trace_dt_utc >= start_dt:
                filtered.append(LangfuseTrace(
                    id=t.id,
                    name=t.name,
                    tags=t.tags or [],
                    timestamp=trace_dt_utc,
                    metadata=t.metadata or {},
                    output=t.output
                ))
        return filtered

    def request_scores(self) -> List[Dict[str, Any]]:
        """
        Request all scores from Langfuse via public API.
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
        Build a map: trace_id -> list of score dicts
        """
        trace_scores_map = defaultdict(list)
        for score in scores:
            trace_id = score.get("traceId")
            if trace_id:
                trace_scores_map[trace_id].append(score)
        return trace_scores_map

    def create_data_rows(self,
                         traces: List[LangfuseTrace],
                         trace_scores_map: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Create a list of row dicts for each trace, including conversation,
        retrieved contexts, and any relevant scores.
        """
        allowed_scores = {"user_feedback", "HumanAnswerCorrectness", "HumanActionNeeded"}
        rows = []

        for trace in traces:
            timestamp_str = trace.timestamp.strftime("%Y-%m-%d %H:%M")

            # Extract relevant data
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
            retrieved_contexts_sorted = sorted(
                retrieved_contexts, key=lambda c: c.get("cosine_distance", 9999)
            )[:5]

            context_text = ""
            for i, ctx in enumerate(retrieved_contexts_sorted, start=1):
                link = ctx.get("link", "No link")
                dist = ctx.get("cosine_distance", "N/A")
                context_text += f"[Context {i}]\nLink: {link}\nCosine Distance: {dist}\n\n"

            original_answer = trace.output if trace.output else ""

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
                data_type = sc.get("dataType", "")
                if name in allowed_scores:
                    if data_type == "CATEGORICAL":
                        row[name] = sc.get("stringValue", "")
                    else:
                        row[name] = sc.get("value", "")

            rows.append(row)
        return rows

    def load_traces_as_dataframe(self) -> pd.DataFrame:
        """
        Fetch & build the DataFrame of traces (with relevant fields).
        """
        traces = self.fetch_filtered_traces()
        scores = self.request_scores()
        trace_scores_map = self.build_trace_scores_map(scores)
        rows = self.create_data_rows(traces, trace_scores_map)
        df = pd.DataFrame(rows)

        preferred_cols = [
            "ID", "Timestamp", "User Question", "Conversation History",
            "Retrieved Context", "Model Thoughts", "Answer", "Expected Answer",
            "Name", "Tags", "user_feedback", "HumanAnswerCorrectness", "HumanActionNeeded"
        ]
        existing_cols = [c for c in preferred_cols if c in df.columns]
        df = df.reindex(columns=existing_cols + [c for c in df.columns if c not in existing_cols])
        return df

    def export_to_csv(self, df: pd.DataFrame, file_name: str = OUTPUT_FILE) -> None:
        """
        Utility to export the final DataFrame to CSV if you want a local record.
        """
        df.to_csv(file_name, index=False)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        Return all existing datasets from Langfuse via public API.
        (If the client sdk has list function, you can use that directly.)
        """
        resp = requests.get(
            f"{self.host}/api/public/datasets",
            auth=HTTPBasicAuth(self.public_key, self.secret_key),
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    def get_or_create_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Look for a dataset by name; create it if not found, return dict with (id, name).
        """
        existing = [ds for ds in self.list_datasets() if ds.get("name") == dataset_name]
        if existing:
            return existing[0]

        # Create using the python client
        client = self._init_client()
        created = client.datasets.create(name=dataset_name)
        return {"id": created["id"], "name": created["name"]}

    def create_or_update_dataset_item(
        self,
        dataset_name: str,
        item_id: str,
        input_data: Dict[str, Any],
        expected_answer: str
    ) -> None:
        """
        Use the Python SDK to create (or upsert) a dataset item.
        By specifying the same 'id', repeated calls update the same item.
        """
        client = self._init_client()
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id,
            input=input_data,
            expected_output={"answer": expected_answer},
            metadata={}
        )
