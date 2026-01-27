import logging
import time
import os
from typing import Dict, Any, Optional
from app.agents.router import RouterAgent
from app.agents.reasoning import CSVReasoningAgent
from app.agents.retriever import CSVRetrieverAgent
from app.agents.answer import AnswerAgent
from app.agents.file_selector import FileSelectorAgent
from app.agents.summary import SummaryAgent
from app.services.registry import file_registry
from app.services.ingestion import ingestion_service
from app.engines.vector_engine import vector_engine
import pandas as pd

logger = logging.getLogger(__name__)

class OrchestrationService:
    """
    Controls the flow: Router -> Reasoning -> Retriever -> Answer.
    """
    def __init__(self):
        self.router = RouterAgent()
        self.reasoning = CSVReasoningAgent()
        self.retriever = CSVRetrieverAgent()
        self.answer = AnswerAgent()
        self.summary_agent = SummaryAgent()
        self.file_selector = FileSelectorAgent()

    def register_data(self, file_path: str):
        """
        Ingests and profiles a file, then adds it to the global registry.
        """
        logger.info(f"Registering file: {file_path}")
        data_bundle = ingestion_service.load_data(file_path)
        df = data_bundle["df"]
        schema_context = data_bundle["semantic_context"]
        
        # Generate summary
        if isinstance(df, dict):
            first_sheet = next(iter(df.values()))
            sample_data = first_sheet.head(5).to_markdown()
        else:
            sample_data = df.head(5).to_markdown()

        file_summary = self.summary_agent.run({
            "schema_context": schema_context,
            "sample_data": sample_data
        })
        
        file_registry.add_file(
            file_path=file_path,
            summary=file_summary,
            row_count=data_bundle["row_count"],
            data_type=data_bundle.get("type", "csv"),
            schema_context=schema_context
        )
        return file_summary

    def run_pipeline(self, query: str, file_path: str, index_name: str) -> Dict[str, Any]:
        """
        Executes the full agentic RAG pipeline.
        """
        start_time = time.time()
        logger.info(f"=== Starting Pipeline for Query: {query} ===")

        # 1. Ingest Data
        logger.info(f"[1/5] Ingesting: {os.path.basename(file_path)}")
        
        # Check registry for cached info
        cached_info = file_registry.get_file_info(file_path)
        
        if cached_info:
            logger.info("Using cached file info from registry.")
            df_data = ingestion_service.load_data(file_path) # Need actual DF for retrieval
            df = df_data["df"]
            schema_context = cached_info["schema_context"]
            file_summary = cached_info["summary"]
            row_count = cached_info["row_count"]
            data_type = cached_info["type"]
        else:
            data_bundle = ingestion_service.load_data(file_path)
            df = data_bundle["df"]
            schema_context = data_bundle["semantic_context"]
            row_count = data_bundle["row_count"]
            data_type = data_bundle.get("type", "csv")
            
            # Generate summary if not in registry
            if isinstance(df, dict):
                first_sheet = next(iter(df.values()))
                sample_data = first_sheet.head(5).to_markdown()
            else:
                sample_data = df.head(5).to_markdown()

            file_summary = self.summary_agent.run({
                "schema_context": schema_context,
                "sample_data": sample_data
            })
            
        logger.info(f"Ready. Type: {data_type}, Total Rows: {row_count}")
        
        # 2. Routing
        logger.info(f"[2/5] Routing query...")
        route_start = time.time()
        route_info = self.router.run({"query": query})
        question_type = route_info["question_type"]
        engine_type = route_info["engine"]
        logger.info(f"Route: {question_type} -> {engine_type} ({time.time() - route_start:.2f}s)")

        # 3. Reasoning (Intent)
        logger.info(f"[3/5] Reasoning (Intent Extraction)...")
        reason_start = time.time()
        intent = self.reasoning.run({
            "query": query,
            "schema_context": schema_context,
            "file_summary": file_summary # Injecting file context
        })
        logger.info(f"Intent extracted in {time.time() - reason_start:.2f}s")

        # 4. Retrieval (Deterministic or Semantic)
        logger.info(f"[4/5] Retrieving from {engine_type}...")
        retrieve_start = time.time()
        retrieved_data = self.retriever.run({
            "query": query,
            "intent": intent,
            "df": df,
            "index_name": index_name,
            "engine_type": engine_type
        })
        logger.info(f"Data retrieved in {time.time() - retrieve_start:.2f}s")

        # 5. Answering
        logger.info(f"[5/5] Synthesizing Answer...")
        answer_start = time.time()
        answer = self.answer.run({
            "query": query,
            "retrieved_data": retrieved_data,
            "intent": intent,
            "file_summary": file_summary # Injecting file context
        })
        logger.info(f"Answer synthesized in {time.time() - answer_start:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"=== Pipeline Finished in {total_time:.2f}s ===")

        return {
            "answer": answer,
            "question_type": question_type,
            "intent": intent,
            "retrieved_data": retrieved_data,
            "file_summary": file_summary,
            "metadata": {
                "execution_time": total_time,
                "data_type": data_type,
                "row_count": row_count
            }
        }

    def run_multi_file_pipeline(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Determines the correct file from the registry if not provided, then runs the pipeline.
        """
        if not file_path:
            logger.info("No file_path provided. Consulting FileSelectorAgent...")
            all_summaries = file_registry.get_all_summaries()
            if not all_summaries:
                return {"answer": "I don't have any files to analyze. Please upload one first.", "status": "error"}
            
            selected_files = self.file_selector.run({
                "query": query,
                "file_summaries": all_summaries
            })
            
            if not selected_files:
                return {"answer": "I couldn't identify which file you're referring to from the available data. Could you specify which file (e.g., 'In the sales file...') or ask about the available files?", "status": "error"}
            
            file_path = selected_files[0] # Pick the first relevant one for now
            logger.info(f"FileSelectorAgent selected: {file_path}")

        return self.run_pipeline(query, file_path, os.path.basename(file_path))

# Singleton
orchestrator = OrchestrationService()
