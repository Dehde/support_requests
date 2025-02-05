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
from pprint import pprint

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
        
        # Load traces and scores
        traces = self.fetch_filtered_traces()
        scores = self.request_scores()
        scores_map = self.build_trace_scores_map(scores)
        data_rows = self.create_data_rows(traces, scores_map)
        
        # Store total traces count before deduplication
        total_traces = len(data_rows)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Only drop duplicates if the column exists
        if 'User Question' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp', ascending=True)
            df = df.drop_duplicates(subset=['User Question'], keep='first')
        
        # Store both counts in the DataFrame metadata
        df.attrs['total_traces_count'] = total_traces
        df.attrs['unique_traces_count'] = len(df)
        
        # Analyze and print statistics
        self._analyze_scores(df)
        
        print(f"Total traces before deduplication: {total_traces}")
        print(f"Unique traces after deduplication: {len(df)}")
        print("Exiting function: load_traces_as_dataframe")
        return df

    def _analyze_scores(self, df: pd.DataFrame) -> None:
        """Analyze and print score statistics."""
        print(f"\nTotal traces loaded: {len(df)}")
        
        # Find boolean score columns
        bool_columns = [col for col in df.columns if col in [
            'context_added', 'user_question_needs_clarification', 
            'llm_failure', 'context_missing', 'retrieval_failure'
        ]]
        
        if bool_columns:
            print(f"\nFound boolean score columns: {bool_columns}")
            print("\nAnalyzing boolean scores...")
            
            for col in bool_columns:
                true_count = df[col].eq(True).sum()
                false_count = df[col].eq(False).sum()
                null_count = df[col].isna().sum()
                
                print(f"\n{col}:")
                print(f"  True values: {true_count}")
                print(f"  False values: {false_count}")
                print(f"  Null values: {null_count}")
                
                if true_count > 0:
                    print("\n  Examples of True values:")
                    true_examples = df[df[col] == True]['ID'].head(3)
                    for trace_id in true_examples:
                        print(f"    Trace ID: {trace_id}, Value: {df[df['ID'] == trace_id][col].iloc[0]}")
        
        # Count traces with at least one score
        traces_with_scores = df[bool_columns].notna().any(axis=1).sum()
        print(f"\nTraces with at least one score: {traces_with_scores}")
        
        if traces_with_scores > 0:
            print("\nFirst few traces with scores:")
            traces_with_any_score = df[df[bool_columns].notna().any(axis=1)]
            print(traces_with_any_score[['ID', 'Timestamp'] + bool_columns].head().to_string())

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
        all_scores = []
        page = 1
        
        while True:
            url = f"{self.host}/api/public/scores"
            resp = requests.get(
                url,
                auth=HTTPBasicAuth(self.public_key, self.secret_key),
                headers={"Content-Type": "application/json"},
                params={"page": page, "limit": 100}  # increased from 50 to 100 per page
            )
            resp.raise_for_status()
            
            scores = resp.json().get("data", [])
            if not scores:  # No more scores to fetch
                break
            
            print(f"Fetched {len(scores)} score(s) for page={page}")
            all_scores.extend(scores)
            page += 1
        
        print(f"Total scores fetched: {len(all_scores)}")
        print("Exiting function: request_scores")
        return all_scores

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
    def create_data_rows(self, traces: List[TraceWithDetails], trace_scores_map: Dict[str, List[Dict[str, Any]]]) -> List[dict]:
        print("Entered function: create_data_rows")
        data_rows = []
        
        for trace in traces:
            ts_str = trace.timestamp.strftime("%Y-%m-%d %H:%M:%S")  # Changed here
            user_q = trace.metadata.get("user_question", "")
            model_thoughts = trace.metadata.get("model_thoughts", "")
            conv_entries = trace.metadata.get("conversation_history") or []
            conversation_history = "\n".join(
                f"{e.get('role', 'unknown')}: {e.get('content', '')}"
                for e in conv_entries
            )
            
            # Process retrieved contexts
            retrieved_contexts = trace.metadata.get("retrieved_contexts", []) if trace.metadata else []
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
            
            # Process scores
            scores = trace_scores_map.get(trace.id, [])
            for score in scores:
                name = score.get("name", "")
                value = score.get("value")
                comment = score.get("comment")
                
                if comment and comment != "":
                    row[name] = comment
                elif value is not None:
                    row[name] = str(value)
                else:
                    row[name] = ""
            
            data_rows.append(row)
        
        print("Exiting function: create_data_rows")
        return data_rows

    def get_latest_scores_by_name(self, scores: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        scores_by_name = defaultdict(list)
        for sc in scores:
            scores_by_name[sc.get('name', '')].append(sc)
        
        latest_scores = {}
        for name, scores in scores_by_name.items():
            # Sort by timestamp descending
            sorted_scores = sorted(scores, key=lambda x: x.get('timestamp', ''), reverse=True)
            latest = sorted_scores[0]
            
            latest_scores[name] = latest
        
        return latest_scores

    def process_score_value(self, score: Dict[str, Any]) -> Any:
        data_type = score.get('dataType', 'CATEGORICAL')
        
        if data_type == "BOOLEAN":
            raw_val = score.get("value")
            if raw_val is None:
                val = None
            elif isinstance(raw_val, bool):
                val = raw_val
            elif isinstance(raw_val, (int, float)):
                val = bool(raw_val)
            elif isinstance(raw_val, str):
                val = raw_val.lower() == "true"
            else:
                val = None
        elif data_type == "NUMERIC":
            val = score.get("value")
        else:  # CATEGORICAL or fallback
            val = score.get("stringValue", str(score.get("value", "")))
        
        return val

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
    import streamlit as st
    from collections import defaultdict

    # Initialize client
    client = LangfuseClient(
        public_key=st.secrets["LANGFUSE_PUBLIC_KEY"],
        secret_key=st.secrets["LANGFUSE_SECRET_KEY"],
        host=st.secrets["LANGFUSE_HOST"]
    )

    # Load traces
    print("\nLoading traces...")
    df = client.load_traces_as_dataframe()