from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    file_path: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    context: List[str]
