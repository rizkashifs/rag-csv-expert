import logging
import time
from typing import Dict, Any
from app.agents.router import RouterAgent
from app.agents.reasoning import CSVReasoningAgent
from app.agents.retriever import CSVRetrieverAgent
from app.agents.answer import AnswerAgent
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

    def run_pipeline(self, query: str, file_path: str, index_name: str) -> Dict[str, Any]:
        """
        Executes the full agentic RAG pipeline.
        """
        start_time = time.time()
        logger.info(f"Starting pipeline for query: {query}")

        # 1. Ingest Data
        logger.info("Stage 1: Ingesting Data...")
        data_bundle = ingestion_service.read_csv(file_path)
        df = data_bundle["df"]
        schema_context = data_bundle["semantic_context"]
        logger.info(f"Ingestion complete. Rows: {len(df)}")
        
        # 2. Routing
        logger.info("Stage 2: Routing query...")
        route_info = self.router.run({"query": query})
        question_type = route_info["question_type"]
        engine_type = route_info["engine"]
        logger.info(f"Route determined: {question_type} using {engine_type}")

        # 3. Reasoning (Intent)
        logger.info("Stage 3: Reasoning (Intent Extraction)...")
        reason_start = time.time()
        intent = self.reasoning.run({
            "query": query,
            "schema_context": schema_context
        })
        logger.info(f"Reasoning complete in {time.time() - reason_start:.2f}s. Intent: {intent}")

        # 4. Retrieval (Deterministic or Semantic)
        logger.info(f"Stage 4: Retrieving data from {engine_type}...")
        retrieve_start = time.time()
        retrieved_data = self.retriever.run({
            "query": query,
            "intent": intent,
            "df": df,
            "index_name": index_name,
            "engine_type": engine_type
        })
        logger.info(f"Retrieval complete in {time.time() - retrieve_start:.2f}s.")

        # 5. Answering
        logger.info("Stage 5: Synthesizing Answer...")
        answer_start = time.time()
        answer = self.answer.run({
            "query": query,
            "retrieved_data": retrieved_data,
            "intent": intent
        })
        logger.info(f"Answer synthesized in {time.time() - answer_start:.2f}s.")

        total_time = time.time() - start_time
        logger.info(f"Pipeline finished total time: {total_time:.2f}s")

        return {
            "answer": answer,
            "question_type": question_type,
            "intent": intent,
            "retrieved_data": str(retrieved_data)[:500] # Truncated for response logs
        }

# Singleton
orchestrator = OrchestrationService()
