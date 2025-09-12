from pydantic import BaseModel
from typing import Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    tool_calls: list = []
    error: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    message: str