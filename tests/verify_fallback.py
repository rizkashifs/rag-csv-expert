
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from app.services.orchestration import orchestrator

class TestFallback(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({"Sales": [100, 200, 300], "Region": ["North", "South", "North"]})
        self.schema_context = "Columns: Sales (int), Region (str)"
        self.file_summary = "Sales data"
        
    @patch('app.services.orchestration.ingestion_service')
    @patch('app.services.orchestration.file_registry')
    @patch('app.agents.router.RouterAgent.run')
    def test_router_crash_fallback(self, mock_router_run, mock_registry, mock_ingestion):
        # 1. Setup Data Mock
        mock_ingestion.load_data.return_value = {
            "df": self.df,
            "semantic_context": self.schema_context,
            "row_count": 3,
            "type": "csv",
            "text_heavy": False,
            "sample_data": ""
        }
        mock_registry.get_file_info.return_value = None # Force load_data
        
        # 2. Simulate Router Crash
        mock_router_run.side_effect = Exception("Simulated Router Crash")
        
        # 3. Run Pipeline
        # Query: "Show total Sales" -> Keyword Engine should catch "total" (sum) and "Sales".
        result = orchestrator.run_pipeline("Show total Sales", "dummy.csv", "dummy_index")
        
        # 4. Assertions
        print("Pipeline Result Class:", result.get("question_type"))
        print("Pipeline Result Answer:", result.get("answer"))
        
        # Should NOT be error_fallback if keyword engine worked
        self.assertNotEqual(result["question_type"], "error_fallback")
        self.assertEqual(result["question_type"], "sql_engine")
        
        # Check Intent
        intent = result["intent"]
        self.assertEqual(intent.get("operation"), "sum")
        self.assertIn("Sales", intent.get("columns", []))
        
        # Check Answer contains result (sum = 600)
        # The AnswerAgent might format it, but let's check retrieved_data
        retrieved = result["retrieved_data"]
        # retrieved_data from SQL engine is usually {"data": ..., "row_count": ...}
        # In orchestration, it calls retriever -> sql_engine.execute
        # Let's verify data
        print("Retrieved Data:", retrieved)
        self.assertEqual(retrieved["data"], {"Sales": 600})

if __name__ == '__main__':
    unittest.main()
