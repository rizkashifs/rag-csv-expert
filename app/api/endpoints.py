import shutil
import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.ingestion import ingestion_service
from app.engines.vector_engine import vector_engine
from app.services.retrieval import retrieve_csv
from app.services.orchestration import orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Uploading file: {file.filename}")
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are allowed")
    
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Ingest & Profile
    logger.info(f"Ingesting and profiling: {file.filename}")
    data_bundle = ingestion_service.load_data(file_path)
    df_data = data_bundle["df"]
    
    # 2. Index for Vector Search
    logger.info("Preparing documents for vector indexing...")
    documents = []
    
    if data_bundle["type"] == "excel":
        for sheet_name, df in df_data.items():
            # Sample from each sheet
            sample = df.head(50)
            if not sample.empty:
                # Add sheet name context to each document
                sheet_docs = sample.to_markdown(index=False).split('\n')
                documents.extend([f"[Sheet: {sheet_name}] {doc}" for doc in sheet_docs if doc.strip()])
    else:
        # CSV handling
        sample_df = df_data.head(100)
        documents = [doc for doc in sample_df.to_markdown(index=False).split('\n') if doc.strip()]
    
    logger.info(f"Creating vector index with {len(documents)} document chunks...")
    try:
        vector_engine.create_index(documents, index_name=file.filename)
    except Exception as e:
        logger.error(f"Vector Indexing Error: {e}")
        raise HTTPException(status_code=503, detail=f"Embedding/Indexing failed: {str(e)}")
    
    # 3. Register with Orchestrator (Generates Summary & adds to Registry)
    file_summary = orchestrator.register_data(file_path)
    
    return {
        "message": "File processed and registered successfully",
        "file_path": file_path,
        "type": data_bundle["type"],
        "row_count": data_bundle["row_count"],
        "summary": file_summary,
        "indexed_chunks": len(documents)
    }

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    logger.info(f"Query request received. Target file: {request.file_path or 'Auto-Detect'}")
    
    result = retrieve_csv(request.query, request.file_path)
    logger.info("Query processing complete.")
    
    # Ensure context is a list of strings as expected by QueryResponse
    raw_data = result.get("retrieved_data", [])
    
    if isinstance(raw_data, list):
        # Handle list of strings or list of dicts
        context = [str(item) for item in raw_data]
    elif isinstance(raw_data, dict):
        # Handle deterministic calculation results
        context = [f"{k}: {v}" for k, v in raw_data.items()]
    else:
        context = [str(raw_data)]

    return QueryResponse(
        answer=result["answer"],
        context=context,
        metadata=result.get("metadata")
    )
