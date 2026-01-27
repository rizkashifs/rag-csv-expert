import json
import logging
import re
import pandas as pd
import chardet
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Handles robust CSV and Excel loading, cleaning, and profiling.
    Supports multi-sheet Excel files.
    """
    def _generate_data_profile(self, df: pd.DataFrame, sheet_name: str = None) -> str:
        """Generates a semantic profile of the DataFrame."""
        profile = {
            "summary": {
                "total_rows": len(df), 
                "total_columns": len(df.columns),
                "sheet_name": sheet_name
            },
            "columns": {}
        }
        for col in df.columns:
            col_type = str(df[col].dtype)
            col_data = {"type": col_type, "missing_values": int(df[col].isnull().sum())}
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_data.update({
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else "N/A",
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else "N/A",
                    })
                else:
                    unique_vals = df[col].dropna().unique()
                    col_data["unique_samples"] = [str(x) for x in unique_vals[:5]]
            except Exception:
                col_data["info"] = "Could not generate stats"
            profile["columns"][col] = col_data
        return json.dumps(profile, indent=2)

    def _format_profile_for_llm(self, profile_json: str) -> str:
        """Formats the profile for LLM context."""
        data = json.loads(profile_json)
        sheet_info = f" (Sheet: {data['summary']['sheet_name']})" if data['summary']['sheet_name'] else ""
        lines = [f"Dataset{sheet_info}: {data['summary']['total_rows']} rows, {data['summary']['total_columns']} cols"]
        for col, info in data["columns"].items():
            line = f"- {col} ({info['type']}): "
            if "min" in info:
                line += f"Range [{info['min']} to {info['max']}]"
            elif "unique_samples" in info:
                line += f"Samples: {', '.join(info['unique_samples'])}"
            lines.append(line)
        return "\n".join(lines)

    def load_data(self, file_path: str) -> Dict[str, Any]:
        """
        Loads CSV or Excel and returns structured data + metadata.
        If Excel, handles multiple sheets.
        """
        try:
            is_excel = file_path.endswith(('.xlsx', '.xls'))
            
            if is_excel:
                logger.info(f"Loading Excel file: {file_path}")
                sheets = pd.read_excel(file_path, sheet_name=None)
                all_profiles = []
                cleaned_sheets = {}
                
                for name, df in sheets.items():
                    # Cleaning
                    df.dropna(how='all', inplace=True)
                    df.columns = [re.sub(r'[^\w\s]', '', str(col).strip()) for col in df.columns]
                    df.fillna("N/A", inplace=True)
                    cleaned_sheets[name] = df
                    
                    profile_json = self._generate_data_profile(df, sheet_name=name)
                    all_profiles.append(self._format_profile_for_llm(profile_json))
                
                return {
                    "df": cleaned_sheets, # Dictionary of DataFrames
                    "type": "excel",
                    "sheets": list(cleaned_sheets.keys()),
                    "semantic_context": "\n\n".join(all_profiles),
                    "row_count": sum(len(d) for d in cleaned_sheets.values())
                }
            else:
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
                profile_json = self._generate_data_profile(df)
                semantic_context = self._format_profile_for_llm(profile_json)
                
                return {
                    "df": df, # Single DataFrame
                    "type": "csv",
                    "schema": df.columns.tolist(),
                    "semantic_context": semantic_context,
                    "row_count": len(df)
                }
        except Exception as e:
            logger.error(f"Ingestion Error: {e}")
            raise

    def read_csv(self, file_path: str) -> Dict[str, Any]:
        """Legacy compatibility wrapper"""
        return self.load_data(file_path)

# Singleton
ingestion_service = IngestionService()
