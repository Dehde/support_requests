# trace_model.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class LangfuseTrace:
    """
    Represents a single trace from Langfuse, including:
      - ID, Name, and Tags
      - Timestamp (UTC)
      - metadata with user question, conversation history, model thoughts, etc.
      - output (the model's final answer or generation)
    """
    id: str
    name: Optional[str]
    tags: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    output: Optional[str] = None
