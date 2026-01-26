import pandas as pd
import json
from typing import Dict, Any

def generate_data_profile(df: pd.DataFrame) -> str:
    """
    Generates a semantic profile of the DataFrame:
    - Column names and types
    - Basic statistics (min, max, mean for numeric)
    - Sample unique values for categorical
    - Missing value counts
    """
    profile = {
        "summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns)
        },
        "columns": {}
    }
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        col_data = {
            "type": col_type,
            "missing_values": int(df[col].isnull().sum()),
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else "N/A",
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else "N/A",
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else "N/A",
            })
        else:
            # Sample unique values for categorical/text
            unique_vals = df[col].dropna().unique()
            col_data["unique_samples"] = [str(x) for x in unique_vals[:5]]
            col_data["total_unique"] = len(unique_vals)
            
        profile["columns"][col] = col_data
        
    return json.dumps(profile, indent=2)

def format_profile_for_llm(profile_json: str) -> str:
    """
    Converts the profile JSON into a readable string for the LLM context.
    """
    data = json.loads(profile_json)
    lines = [f"Dataset Overview: {data['summary']['total_rows']} rows, {data['summary']['total_columns']} columns\n"]
    lines.append("Column Profiles:")
    
    for col, info in data["columns"].items():
        line = f"- {col} ({info['type']}): "
        if "min" in info:
            line += f"Range [{info['min']} to {info['max']}], Mean: {info['mean']:.2f}" if isinstance(info['mean'], (int, float)) else f"Range [{info['min']} to {info['max']}]"
        elif "unique_samples" in info:
            line += f"{info['total_unique']} unique values. Samples: {', '.join(info['unique_samples'])}"
        
        if info["missing_values"] > 0:
            line += f" | {info['missing_values']} missing values"
        
        lines.append(line)
        
    return "\n".join(lines)
