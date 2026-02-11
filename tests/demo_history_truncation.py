"""
Demonstration of the history truncation utility.
"""
from app.utils.history_utils import truncate_history

# Example: Full conversation history with 7 turns
full_history = [
    {"user": "Show me all the data", "assistant": "Here are 1000 rows..."},
    {"user": "What is the total sales?", "assistant": "Total sales is $50,000"},
    {"user": "Group by region", "assistant": "Grouped data: North: $20k, South: $30k"},
    {"user": "What about the average?", "assistant": "Average is $500 per transaction"},
    {"user": "Show top 5 products", "assistant": "Top products: Product A, B, C, D, E"},
    {"user": "Filter by date > 2023", "assistant": "Filtered to 500 rows"},
    {"user": "Sort by revenue descending", "assistant": "Sorted data shown"},
]

print("Full conversation history has", len(full_history), "turns")
print("\n" + "="*60)
print("BEFORE: Traditional approach (last 5 turns with assistant)")
print("="*60)
traditional = "\n".join(
    [f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}" 
     for turn in full_history[-5:]]
)
print(traditional)
print(f"\nCharacter count: {len(traditional)}")

print("\n" + "="*60)
print("AFTER: New approach (last 5 user queries only)")
print("="*60)
optimized = truncate_history(full_history, max_user_turns=5)
print(optimized)
print(f"\nCharacter count: {len(optimized)}")

print("\n" + "="*60)
print("SAVINGS")
print("="*60)
print(f"Reduced from {len(traditional)} to {len(optimized)} characters")
print(f"Savings: {len(traditional) - len(optimized)} characters ({100 * (1 - len(optimized)/len(traditional)):.1f}%)")
print("\nBenefits:")
print("✓ Reduced token usage for LLM context")
print("✓ Only relevant user queries (no verbose assistant responses)")
print("✓ Cleaner, more focused context for routing decisions")
