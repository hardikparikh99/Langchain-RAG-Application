from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- API Models ---

class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    status: str
    message: str
    chunk_count: Optional[int] = 0

class DocumentDeleteResponse(BaseModel):
    document_id: str
    status: str
    message: str

class QueryRequest(BaseModel):
    query: str
    conversation_id: str
    k: int = 3  # Default value from config

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[ConversationMessage]