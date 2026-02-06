"""
Centralized routing keyword patterns for keyword-based intent detection.
"""

SIMPLE_INTENT_PATTERNS = {
    "sum": [r"\bsum\b", r"\btotal\b"],
    "avg": [r"\baverage\b", r"\bavg\b", r"\bmean\b"],
    "count": [r"\bcount\b", r"\bhow many\b"],
    "max": [r"\bmax\b", r"\bhighest\b", r"\blargest\b"],
    "min": [r"\bmin\b", r"\blowest\b", r"\bsmallest\b"],
    "correlation": [r"\bcorrelation\b", r"\bcorr\b", r"\brelationship\b", r"\bcompare\b"],
}

PROFILE_KEYWORDS = [
    "schema",
    "columns",
    "fields",
    "profile",
    "overview",
    "summary",
]
