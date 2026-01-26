import pandas as pd
import chardet
import re
import logging
from typing import Tuple, List, Dict, Any
from app.services.data_profiler import generate_data_profile, format_profile_for_llm

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Handles robust CSV loading, cleaning, and profiling.
    """
    def read_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Loads CSV with robust detection and returns structured data + metadata.
        """
        try:
            # Detect encoding
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                encoding = result['encoding'] or 'utf-8'

            # Robust read
            df = pd.read_csv(file_path, sep=None, engine='python', encoding=encoding)
            
            # Cleaning
            df.dropna(how='all', inplace=True)
            df.columns = [re.sub(r'[^\w\s]', '', col.strip()) for col in df.columns]
            df.fillna("N/A", inplace=True)

            # Profiling
            profile_json = generate_data_profile(df)
            semantic_context = format_profile_for_llm(profile_json)
            
            return {
                "df": df,
                "schema": df.columns.tolist(),
                "semantic_context": semantic_context,
                "row_count": len(df)
            }
        except Exception as e:
            logger.error(f"Ingestion Error: {e}")
            raise

# Singleton
ingestion_service = IngestionService()
