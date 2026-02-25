# Engine Output Contract

This document describes every possible output shape returned by `csv_engine` (`SQLEngine`) and `text_engine` (`TextEngine`).

Both engines share the **same top-level contract** so that all downstream logic (the `AnswerAgent`, API responses, UI rendering) can handle them identically.

---

## CSV vs. Excel (Multi-Sheet) Core Differences

While both engines return the same JSON structure, the **granularity** of the data changes when moving from a single CSV to a multi-sheet Excel workbook.

| Feature | CSV (Single File) | Excel (Multi-Sheet) |
|---|---|---|
| **Computation Scope** | Global (entire file) | Local per sheet |
| **`sheet` Metadata** | Absent (or "default") | Always present |
| **Aggregations** | Returns 1 result per query | Returns **N results** (1 per sheet) |
| **Identity** | Rows are raw | Rows injected with `sheet` source |

### The "Federated Search" Principle
When an Excel file is provided, the engine treats it as a collection of independent tables. A query like *"What is the average price?"* will trigger the engine to compute the average on **every tab** that has a "Price" column. This enables "Federated" responses like *"The average in Tab A is $10, while in Tab B it is $15."*

---

## Top-Level Shape

Both engines **always** return one of two top-level shapes:

```python
# SUCCESS
{"relevant_rows": [ ... list of row dicts ... ]}

# REFUSAL / FAILURE  (via RefusalAgent)
{"relevant_rows": [{"_summary": "I need a bit more detail...", "should_ask_user": True}]}
```

> **Rule:** Always read `relevant_rows`. If the list is empty or the first element contains `should_ask_user: True`, treat it as a failure / clarification request.

---

## Reserved Keys

These keys appear on rows produced by the engines (not raw CSV columns). Never assume they are absent — always check before rendering.

| Key | Type | Engines | Meaning |
|---|---|---|---|
| `_summary` | `str` | both | Human-readable one-line description of the row |
| `should_ask_user` | `bool` | both | `True` → this is a refusal/clarification row, not data |
| `column` | `str` | csv_engine (profile) | Name of the profiled column |
| `stats` | `str` | csv_engine (profile) | Pipe-separated stat string for the column |
| `count` | `str` | csv_engine (count) | Formatted count string |
| `correlation` | `dict` | csv_engine (corr) | Nested correlation matrix dict |
| `sheet` | `str` | both (multi-sheet) | Sheet name when processing Excel with multiple sheets |
| `_matched_columns` | `list[str]` | text_engine | Columns that were searched in text matching |
| `_keywords` | `list[str]` | text_engine | Keywords extracted and used in the search |

---

## csv_engine — All Output Variants

Triggered by the `SQL_ENGINE` route. Operates with Pandas on the full DataFrame.

---

### 1. Tabular Rows — `filter` / `none` / `group_by`

The most common case. Raw DataFrame rows serialised as dicts, limited to `intent.limit` (default 1000).

**CSV (Single Sheet):**
```python
{
    "relevant_rows": [
        {"EmployeeID": 1, "Name": "Alice", "Salary": 50000, "Department": "HR"},
        {"EmployeeID": 2, "Name": "Bob",   "Salary": 60000, "Department": "Eng"},
        # ... up to `limit` rows
    ]
}
```

**Excel (Multi-Sheet):**
Each row includes the derived `sheet` metadata to identify its origin.
```python
{
    "relevant_rows": [
        {"EmployeeID": 1, "Name": "Alice", "sheet": "Q1_Active"},
        {"EmployeeID": 2, "Name": "Bob",   "sheet": "Q1_Active"},
        {"EmployeeID": 5, "Name": "Carol", "sheet": "Q2_Archived"},
    ]
}
```

---

### 2. Count Scalar

`operation = "count"` — always a single row.

```python
{
    "relevant_rows": [
        {
            "count":    "Count: 42",
            "_summary": "Count of rows where Department is Eng: 42"
        }
    ]
}
```

When no filter is active:
```python
{
    "relevant_rows": [
        {
            "count":    "Count: 200",
            "_summary": "Count of rows: 200"
        }
    ]
}
```

---

### 3. Sum / Avg / Min / Max Scalar

`operation = "sum" | "avg" | "min" | "max"` — **one row per column** that was computed.

**CSV (Single Sheet):**
Returns one aggregation per requested column.
```python
{
    "relevant_rows": [
        {
            "Salary": "Average of Salary: 55000.0",
            "_summary": "Average of Salary: 55000.0"
        }
    ]
}
```

**Excel (Multi-Sheet) — Federated Aggregation:**
The engine calculates the average for **every sheet** that contains the target column. This allows the LLM to compare values across categories (e.g. Regions).
```python
{
    "relevant_rows": [
        {
            "Salary": "Average of Salary: 42000.0",
            "sheet": "North_Region",
            "_summary": "[North_Region] Average of Salary: 42000.0"
        },
        {
            "Salary": "Average of Salary: 68000.0",
            "sheet": "South_Region",
            "_summary": "[South_Region] Average of Salary: 68000.0"
        }
    ]
}
```

---

### 4. Profile / Summary Scalar

`operation = "profile"` — **one row per column** profiled, with pipe-separated statistics.

```python
{
    "relevant_rows": [
        {
            "column":   "JoinDate",
            "stats":    "Count: 200 | Unique: 198 | Min Date: 2018-01-15 | Max Date: 2023-11-30",
            "_summary": "Summary for 'JoinDate': Count: 200 | Unique: 198 | Min Date: 2018-01-15 | Max Date: 2023-11-30"
        },
        {
            "column":   "Salary",
            "stats":    "Count: 200 | Unique: 45 | Min: 30000.0 | Max: 120000.0 | Avg: 67500.0",
            "_summary": "Summary for 'Salary': Count: 200 | Unique: 45 | Min: 30000.0 | Max: 120000.0 | Avg: 67500.0"
        },
        {
            "column":   "Name",
            "stats":    "Count: 200 | Unique: 198",
            "_summary": "Summary for 'Name': Count: 200 | Unique: 198"
        }
    ]
}
```

**Stats rules by data type:**
- **Numeric column** → `Count | Unique | Min | Max | Avg`
- **Date column** → `Count | Unique | Min Date | Max Date`
- **Text column** → `Count | Unique` (no min/max computed)

---

### 5. Correlation Scalar

`operation = "correlation" | "corr"` — always a single row. The `correlation` value is a **nested dict**, not a scalar.

```python
{
    "relevant_rows": [
        {
            "correlation": {
                "Salary": {"Salary": 1.0,  "Bonus": 0.87},
                "Bonus":  {"Salary": 0.87, "Bonus": 1.0}
            },
            "_summary": "Correlation matrix computed."
        }
    ]
}
```

> ⚠️ `correlation` is the only case where a row value is itself a **nested dict**. Handle separately before rendering.

---

### 6. Computation Failure (Unresolvable Scalar)

When a column can't be resolved or the numeric conversion yields no values.

```python
{
    "relevant_rows": [
        {
            "_summary":       "Average could not be computed for Salary.",
            "should_ask_user": True
        }
    ]
}
```

---

### 7. No Rows Matched Filters

When filters are valid but return zero rows from the DataFrame.

```python
{
    "relevant_rows": [
        {
            "_summary":       "No rows matched your filters.",
            "should_ask_user": True
        }
    ]
}
```

---

### 8. Refusal — Filter / Grouping / Total Failure

Triggered when: a filter column is missing, grouping crashes, or the entire computation produces no output. Routed through `RefusalAgent`.

```python
{
    "relevant_rows": [
        {
            "_summary": (
                "I need a bit more detail before I can run this request accurately.\n"
                "Please clarify the following:\n"
                "- Filter column 'Dept' was not found in the dataset.\n\n"
                "Here is the dataset profile to help you choose:\n\n"
                "default: EmployeeID, Name, Salary, Department, JoinDate"
            ),
            "should_ask_user": True
        }
    ]
}
```

---

## text_engine — All Output Variants

Triggered by the `TEXT_TABLE_RAG` route. Uses grep/pandas text matching — no embeddings.

---

### 9. Matched Text Rows — Success

Each matched row contains **all original CSV columns** plus three engine-added metadata keys.

```python
{
    "relevant_rows": [
        {
            # All original CSV columns for this row:
            "EmployeeID": 1234,
            "Name":       "Alice",
            "Department": "Engineering",
            "Comments":   "She is a great team player and very proactive.",

            # Engine metadata (always present on text matches):
            "_summary":         "[default] Matched 'great, proactive' — Comments: She is a great team player and very proactive.",
            "_matched_columns": ["Comments", "Bio"],
            "_keywords":        ["great", "proactive"]
        },
        {
            "EmployeeID": 5678,
            "Name":       "Bob",
            "Department": "Sales",
            "Comments":   "Proactive with clients and excels under pressure.",

            "_summary":         "[default] Matched 'great, proactive' — Comments: Proactive with clients and excels under pressure.",
            "_matched_columns": ["Comments", "Bio"],
            "_keywords":        ["great", "proactive"]
        }
        # ... up to `top_k` rows (default 8)
    ]
}
```

**With id_filters scoping** (e.g. "show comments for employee 1234"):
```python
{
    "relevant_rows": [
        {
            "EmployeeID":       1234,
            "Name":             "Alice",
            "Comments":         "She is a great team player and very proactive.",
            "_summary":         "[default] Matched 'comments' — Comments: She is a great team player and very proactive.",
            "_matched_columns": ["Comments"],
            "_keywords":        ["comments"]
        }
    ]
}
```

**With multi-sheet Excel** (extra `sheet` key injected):
```python
{
    "relevant_rows": [
        {
            "Name":             "Alice",
            "Comments":         "Great communicator.",
            "sheet":            "Q1_Employees",
            "_summary":         "[Q1_Employees] Matched 'communicator' — Comments: Great communicator.",
            "_matched_columns": ["Comments"],
            "_keywords":        ["communicator"]
        }
    ]
}
```

---

### 10. No Match / Refusal — text_engine

Any failure (no text columns found, no keyword match, post-filter eliminates all rows) returns the standard refusal shape.

```python
{
    "relevant_rows": [
        {
            "_summary": (
                "I need a bit more detail before I can run this request accurately.\n"
                "Please clarify the following:\n"
                "- No rows matched the keywords ['proactive'] in sheet 'default'.\n"
                "- Should I search in a specific column instead?\n\n"
                "Here is the dataset profile to help you choose:\n\n"
                "default: EmployeeID, Name, Department, Comments"
            ),
            "should_ask_user": True
        }
    ]
}
```

---

## Quick Reference — Identifying Row Types

| Condition | Row type |
|---|---|
| `row.get("should_ask_user") == True` | **Refusal / clarification** — show `_summary` to user |
| `"column" in row and "stats" in row` | **Profile row** — render `column` + `stats` |
| `"correlation" in row` | **Correlation matrix** — render nested dict |
| `"count" in row` | **Count scalar** — render `count` string |
| `"_keywords" in row` | **Text match row** (text_engine) — render full row + `_summary` |
| `"_summary" in row` (none of above) | **Aggregation scalar** (sum/avg/min/max) — render `_summary` |
| None of the above | **Plain tabular row** (filter/select) — render all non-`_` keys |

---

## Recommended Downstream Guard Pattern

```python
result = engine.execute(df, intent)          # or text_engine.execute(df, intent)
rows = result.get("relevant_rows", [])

# --- guard: total empty (should not normally happen) ---
if not rows:
    show_error("No data was returned. Please try again.")
    return

# --- guard: refusal / clarification ---
if rows[0].get("should_ask_user"):
    show_clarification_prompt(rows[0]["_summary"])
    return

# --- render data rows ---
for row in rows:
    if "column" in row and "stats" in row:
        render_profile_row(row["column"], row["stats"])

    elif "correlation" in row:
        render_correlation_matrix(row["correlation"])

    elif "count" in row:
        render_scalar(row["_summary"])

    elif "_keywords" in row:
        render_text_match(row)         # text_engine result

    elif "_summary" in row:
        render_scalar(row["_summary"]) # aggregation (sum/avg/min/max)

    else:
        render_table_row(row)          # plain tabular row
```

---

## Engine Entry Points

| Engine | Class | Singleton | Called by |
|---|---|---|---|
| `csv_engine.py` | `SQLEngine` | `sql_engine` / `csv_engine` | `CSVRetrieverAgent` (`engine_type="sql_engine"`) |
| `text_engine.py` | `TextEngine` | `text_engine` | `CSVRetrieverAgent` (`engine_type="TEXT_TABLE_RAG"`) |
| `vector_engine.py` | `VectorEngine` | `vector_engine` | Not used in the main pipeline (retained for future embedding support) |
