import shutil
import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.ingestion import ingestion_service
from app.engines.vector_engine import vector_engine
from app.services.retrieval import retrieve_csv

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Uploading file: {file.filename}")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Ingest & Profile
    logger.info("Ingesting and profiling CSV...")
    data_bundle = ingestion_service.read_csv(file_path)
    
    # 2. Index for Vector Search
    logger.info("Creating vector index...")
    # Convert whole rows to Markdown for semantic search (Sampling first 100 for speed)
    sample_df = data_bundle["df"].head(100)
    documents = sample_df.to_markdown(index=False).split('\n')
    
    try:
        vector_engine.create_index(documents, index_name=file.filename)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    return {
        "message": "File uploaded and indexed successfully (Note: only first 100 rows indexed)",
        "file_path": file_path,
        "row_count": data_bundle["row_count"]
    }

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    logger.info(f"Query request received for file: {request.file_path}")
    if not request.file_path:
         raise HTTPException(status_code=400, detail="file_path is required for now")
    
    result = retrieve_csv(request.file_path, request.query)
    logger.info("Query processing complete.")
    
    # Ensure context is a list as expected by QueryResponse
    context = result.get("retrieved_data", [])
    if isinstance(context, str):
        context = [context]
    elif not isinstance(context, list):
        context = [str(context)]

    return QueryResponse(
        answer=result["answer"],
        context=context
    )
