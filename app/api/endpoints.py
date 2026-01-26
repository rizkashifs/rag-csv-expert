from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.document_processing import read_csv, process_file
from app.services.vector_store import create_vector_store
from app.services.retrieval import retrieve_csv
from app.models.ollama_client import invoke_messages
import shutil
import os

router = APIRouter()

from app.services.ingestion import ingestion_service
from app.engines.vector_engine import vector_engine

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Ingest & Profile
    data_bundle = ingestion_service.read_csv(file_path)
    
    # 2. Index for Vector Search
    # Convert whole rows to Markdown for semantic search
    documents = data_bundle["df"].to_markdown(index=False).split('\n')
    vector_engine.create_index(documents, index_name=file.filename)
    
    return {
        "message": "File uploaded and indexed successfully",
        "file_path": file_path,
        "row_count": data_bundle["row_count"]
    }

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not request.file_path:
         raise HTTPException(status_code=400, detail="file_path is required for now")
    
    result = retrieve_csv(request.file_path, request.query)
    
    return QueryResponse(
        answer=result["answer"],
        context=result["context"]
    )
