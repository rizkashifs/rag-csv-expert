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
from app.agents.refusal import RefusalAgent
from app.services.registry import file_registry
from app.services.ingestion import ingestion_service
from app.engines.vector_engine import vector_engine
from app.services.history import get_history, history_service

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
        self.refusal_agent = RefusalAgent()

    def register_data(self, file_path: str, data_bundle: Optional[Dict[str, Any]] = None):
        """
        Ingests and profiles a file, then adds it to the global registry.
        """
        logger.info(f"Registering file: {file_path}")
        data_bundle = data_bundle or ingestion_service.load_data(file_path)
        df = data_bundle["df"]
        schema_context = data_bundle["semantic_context"]
        text_heavy = data_bundle.get("text_heavy", False)
        
        # Generate summary
        sample_data = data_bundle.get("sample_data") or ""
        file_summary = self.summary_agent.run({
            "schema_context": schema_context,
            "sample_data": sample_data
        })
        
        file_registry.add_file(
            file_path=file_path,
            summary=file_summary,
            row_count=data_bundle["row_count"],
            data_type=data_bundle.get("type", "csv"),
            schema_context=schema_context,
            semantic_summary=file_summary,
            text_heavy=text_heavy
        )
        return file_summary

    def run_pipeline(self, query: str, file_path: str, index_name: str, chat_id: Optional[str] = None) -> Dict[str, Any]:
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
            text_heavy = cached_info.get("text_heavy", False)
        else:
            data_bundle = ingestion_service.load_data(file_path)
            df = data_bundle["df"]
            schema_context = data_bundle["semantic_context"]
            row_count = data_bundle["row_count"]
            data_type = data_bundle.get("type", "csv")
            text_heavy = data_bundle.get("text_heavy", False)
            
            # Generate summary if not in registry
            sample_data = data_bundle.get("sample_data") or ""
            file_summary = self.summary_agent.run({
                "schema_context": schema_context,
                "sample_data": sample_data
            })
            
        logger.info(f"Ready. Type: {data_type}, Total Rows: {row_count}")
        
        # 2. Routing
        logger.info(f"[2/5] Routing query...")
        route_start = time.time()
        route_info = self.router.run({
            "query": query,
            "dataset_profile": schema_context,
            "semantic_summary": file_summary,
            "text_heavy": text_heavy,
            "history": get_history(chat_id)
        })
        route = route_info["route"]
        route_schema = route_info.get("schema", {})
        use_routing_agent = route_info.get("use_routing_agent", False)
        logger.info(f"Route selected: {route} ({time.time() - route_start:.2f}s)")

        if route == "REFUSE":
            answer = self.refusal_agent.run({"schema_context": schema_context, "route_schema": route_schema})
            if chat_id:
                history_service.add_turn(chat_id, query, answer)
            return {
                "answer": answer,
                "question_type": "clarify",
                "intent": route_schema or {},
                "retrieved_data": [],
                "file_summary": file_summary,
                "metadata": {
                    "execution_time": time.time() - start_time,
                    "data_type": data_type,
                    "row_count": row_count,
                    "route": route,
                    "use_routing_agent": use_routing_agent,
                    "route_schema": route_schema
                }
            }

        if route == "PROFILE_ONLY":
            answer = (
                "Here is the dataset profile and semantic summary:\n\n"
                f"{schema_context}\n\nSummary:\n{file_summary}"
            )
            if chat_id:
                history_service.add_turn(chat_id, query, answer)
            return {
                "answer": answer,
                "question_type": "profile_only",
                "intent": route_schema or {},
                "retrieved_data": [],
                "file_summary": file_summary,
                "metadata": {
                    "execution_time": time.time() - start_time,
                    "data_type": data_type,
                    "row_count": row_count,
                    "route": route,
                    "use_routing_agent": use_routing_agent,
                    "route_schema": route_schema
                }
            }

        if route == "TEXT_TABLE_RAG":
            logger.info(f"[3/4] Retrieving text chunks from vector index...")
            retrieve_start = time.time()
            retrieved_data = vector_engine.search(query, index_name)
            logger.info(f"Data retrieved in {time.time() - retrieve_start:.2f}s")

            logger.info(f"[4/4] Synthesizing Answer...")
            answer_start = time.time()
            answer = self.answer.run({
                "query": query,
                "retrieved_data": retrieved_data,
                "intent": route_schema or {"operation": "semantic"},
                "file_summary": file_summary
            })
            logger.info(f"Answer synthesized in {time.time() - answer_start:.2f}s")
            total_time = time.time() - start_time
            if chat_id:
                history_service.add_turn(chat_id, query, answer)
            return {
                "answer": answer,
                "question_type": "semantic",
                "intent": route_schema or {"operation": "semantic"},
                "retrieved_data": retrieved_data,
                "file_summary": file_summary,
                "metadata": {
                    "execution_time": total_time,
                    "data_type": data_type,
                    "row_count": row_count,
                    "route": route,
                    "use_routing_agent": use_routing_agent,
                    "route_schema": route_schema
                }
            }

        # SQL/KEYWORD engine route
        logger.info(f"[3/5] Reasoning (Intent Extraction)...")
        reason_start = time.time()
        intent = route_schema or self.reasoning.run({
            "query": query,
            "schema_context": schema_context,
            "file_summary": file_summary
        })
        logger.info(f"Intent extracted in {time.time() - reason_start:.2f}s")

        logger.info(f"[4/5] Retrieving via SQL Engine...")
        retrieve_start = time.time()
        retrieved_data = self.retriever.run({
            "query": query,
            "intent": route_schema or intent,
            "df": df,
            "index_name": index_name,
            "engine_type": "sql_engine"
        })
        logger.info(f"Data retrieved in {time.time() - retrieve_start:.2f}s")

        logger.info(f"[5/5] Synthesizing Answer...")
        answer_start = time.time()
        answer = self.answer.run({
            "query": query,
            "retrieved_data": retrieved_data,
            "intent": route_schema or intent,
            "file_summary": file_summary
        })
        logger.info(f"Answer synthesized in {time.time() - answer_start:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"=== Pipeline Finished in {total_time:.2f}s ===")
        if chat_id:
            history_service.add_turn(chat_id, query, answer)

        return {
            "answer": answer,
            "question_type": "keyword_engine" if route == "KEYWORD_ENGINE" else "sql_engine",
            "intent": route_schema or intent,
            "retrieved_data": retrieved_data,
            "file_summary": file_summary,
            "metadata": {
                "execution_time": total_time,
                "data_type": data_type,
                "row_count": row_count,
                "route": route,
                "use_routing_agent": use_routing_agent,
                "route_schema": route_schema
            }
        }

    def run_multi_file_pipeline(self, query: str, file_path: Optional[str] = None, chat_id: Optional[str] = None) -> Dict[str, Any]:
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

        return self.run_pipeline(query, file_path, os.path.basename(file_path), chat_id=chat_id)

# Singleton
orchestrator = OrchestrationService()
