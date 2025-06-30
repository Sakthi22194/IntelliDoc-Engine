import json
import os

def review_interactions():
    """Review all logged question-answer interactions"""
    log_file = "C:/IntelliDoc/interactions.json"
    
    if not os.path.exists(log_file):
        print("No interactions found.")
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    
    print(f"Found {len(logs)} interactions:")
    for i, log in enumerate(logs, 1):
        print(f"\nInteraction {i}:")
        print(f"Question: {log['question']}")
        print(f"Answer: {log['answer']}")
        print(f"Context: {log['context'][:100]}...")
        print(f"Timestamp: {log['timestamp']}")
        # Simulate manual review (e.g., flag incorrect answers)
        print("Review: [Manually check if answer is correct]")

if __name__ == "__main__":
    review_interactions()