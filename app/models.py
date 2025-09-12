from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from fastapi import UploadFile

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    tool_calls: list = []
    error: Optional[str] = None
    visualization: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    error: str
    message: str

class UploadResponse(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    message: str

class DataSummaryResponse(BaseModel):
    total_rows: int
    total_columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    missing_values: Dict[str, int]
    data_types: Dict[str, str]

class VisualizationResponse(BaseModel):
    type: str
    plotly_json: Optional[Dict] = None
    image_base64: Optional[str] = None
    data_info: Optional[Dict] = None
    error: Optional[str] = None

class ConversationHistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str
    visualization: Optional[Dict] = None

class ConversationHistoryResponse(BaseModel):
    history: List[ConversationHistoryItem]
    total: int