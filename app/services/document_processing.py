import pandas as pd
import os
import re
import logging
import time
import chardet
from typing import List, Dict, Tuple, Any
from pathlib import Path
from app.models.ollama_client import invoke_messages
from app.core.config import settings
from app.services.data_profiler import generate_data_profile, format_profile_for_llm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock for demonstration, should be integrated with actual store
simple_vector_store = []

class MultimodalPromptBuilder:
    def build(self, file_type: str, content: str, file_name: str, user_question: str) -> str:
        return f"File Name: {file_name}\nFile Type: {file_type}\n\nContent:\n{content}\n\nUser Question: {user_question}"

def generate_summary(prompt: str, model_type: str) -> str:
    messages = [{"role": "user", "content": f"Summarize the following data:\n{prompt}"}]
    return invoke_messages(messages)

def classify_csv_type(df: pd.DataFrame) -> str:
    """
    Classifies CSV as NUMERIC or TEXTUAL based on column types and content.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    # If more than 50% of columns are numeric, classify as NUMERIC
    if len(numeric_cols) > len(text_cols):
        return "NUMERIC"
    
    # Check if text columns contain long strings (typical of surveys/claims)
    avg_text_length = 0
    if len(text_cols) > 0:
        sample_text = df[text_cols].head(10).astype(str)
        avg_text_length = sample_text.applymap(len).values.mean()
    
    if avg_text_length > 50:
        return "TEXTUAL"
    
    return "NUMERIC" # Default to numeric if unsure

def read_csv(file_path: str, max_rows: int = 300, **kwargs) -> Tuple[str, List[Dict], str, str]:
    """Read and preprocess CSV for RAG context with robust detection and profiling"""
    try:
        # Detect encoding
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
            encoding = result['encoding'] or 'utf-8'

        # Robust read with delimiter detection
        df = pd.read_csv(file_path, sep=None, engine='python', encoding=encoding)

        # 1. Drop completely empty rows
        df.dropna(how='all', inplace=True)

        # 2. Fill missing values with placeholder
        df.fillna("N/A", inplace=True)

        # 3. Strip whitespace and remove special characters from column names
        df.columns = [re.sub(r'[^\w\s]', '', col.strip()) for col in df.columns]

        # Generate Data Profile for semantic context
        profile_json = generate_data_profile(df)
        semantic_context = format_profile_for_llm(profile_json)
        
        # Classify CSV type
        csv_type = classify_csv_type(df)

        # Truncate only if using Haiku (keeping your safe limit)
        if df.shape[0] > 10000:
            logger.warning(f"CSV truncated from {df.shape[0]} to 10000 rows (safety limit)")
            df = df.head(10000)
        else:
            logger.info(f"CSV loaded with {df.shape[0]} rows and {df.shape[1]} columns") 

        # convert to list for structured storage
        records = df.to_dict('records')

        # Convert DataFrame to Markdown (expert preference for LLMs)
        limited_df = df.head(10) # Show a bit more in the summary
        csv_string = limited_df.to_markdown(index=False)

        logger.info(f"CSV processed: {df.shape[0]} rows, {df.shape[1]} columns, type: {csv_type}")
        return csv_string, records, semantic_context, csv_type

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return "", [], "", "UNKNOWN"


def process_file(file_path: str, user_question: str = "", model_type: str = "sonnet", max_rows: int = 300):
    start_time = time.time()



    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    if file_path.suffix.lower() == '.csv':
        content, records, semantic_context, csv_type = read_csv(str(file_path), model=model_type, max_rows=max_rows)
        if content:
            # Format documentation with Semantic Context for better LLM understanding
            builder = MultimodalPromptBuilder()
            # Inject Data Profile into the content for the summary prompt
            enriched_content = f"### DATA PROFILE (SCHEMA & STATS):\n{semantic_context}\n\n### DATA SAMPLE (MARKDOWN):\n{content}"
            
            csv_prompt = builder.build(
                file_type="csv",
                content=enriched_content,
                file_name=file_path.name,
                user_question=user_question
            )

            # Generate Summary
            try:
                summary = generate_summary(csv_prompt, model_type)
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                return

            response_time = time.time() - start_time
            simple_vector_store.append({
                "file_type": "csv",
                "source": str(file_path),
                "prompt": csv_prompt,
                "summary": summary,
                "content": records,
                "csv_type": csv_type,
                "response_time": response_time,
            })

            return True

    else:
        logger.warning(f"Unsupported file type: {file_path.suffix}")
        return False
