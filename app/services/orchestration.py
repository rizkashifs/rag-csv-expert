from app.agents.router import RouterAgent
from app.agents.reasoning import CSVReasoningAgent
from app.agents.retriever import CSVRetrieverAgent
from app.agents.answer import AnswerAgent
from app.services.ingestion import ingestion_service
from app.engines.vector_engine import vector_engine
import pandas as pd
from typing import Dict, Any

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
        # 1. Ingest Data
        data_bundle = ingestion_service.read_csv(file_path)
        df = data_bundle["df"]
        schema_context = data_bundle["semantic_context"]
        
        # 2. Routing
        route_info = self.router.run({"query": query})
        question_type = route_info["question_type"]
        engine_type = route_info["engine"]

        # 3. Reasoning (Intent)
        intent = self.reasoning.run({
            "query": query,
            "schema_context": schema_context
        })

        # 4. Retrieval (Deterministic or Semantic)
        retrieved_data = self.retriever.run({
            "intent": intent,
            "df": df,
            "index_name": index_name,
            "engine_type": engine_type
        })

        # 5. Answering
        answer = self.answer.run({
            "query": query,
            "retrieved_data": retrieved_data,
            "intent": intent
        })

        return {
            "answer": answer,
            "question_type": question_type,
            "intent": intent,
            "retrieved_data": str(retrieved_data)[:500] # Truncated for response logs
        }

# Singleton
orchestrator = OrchestrationService()
