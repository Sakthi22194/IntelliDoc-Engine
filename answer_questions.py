import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
from datetime import datetime
import os

# Initialize Chroma
chroma_client = chromadb.PersistentClient(path="C:/IntelliDoc/chroma_db")
collection = chroma_client.get_collection(name="intellidoc_embeddings")

# Initialize Hugging Face models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

def save_interaction(question, answer, context):
    """Log question-answer pair to a JSON file"""
    log_entry = {
        "question": question,
        "answer": answer,
        "context": context[:200],  # Limit context size
        "timestamp": datetime.utcnow().isoformat()
    }
    
    log_file = "C:/IntelliDoc/interactions.json"
    
    # Append to file (create if doesn't exist)
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4)
    
    print(f"Saved interaction: {question}")

def answer_question(question):
    """Generate answer for a given question using the knowledge base"""
    # Convert question to embedding
    question_embedding = embedding_model.encode([question], convert_to_numpy=True).tolist()
    
    # Query Chroma for top 5 relevant sentences
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=5
    )
    
    # Combine relevant sentences into context
    context = " ".join(results['documents'][0])
    print(f"Relevant context: {context[:200]}...")
    
    # Generate answer using Hugging Face
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']
    
    print(f"Question: {question}\nAnswer: {answer}")
    
    # Save interaction
    save_interaction(question, answer, context)
    
    return answer

if __name__ == "__main__":
    # Test with a sample question
    question = "What is the revenue in Q4?"  # Adjust based on your PDF
    answer_question(question)