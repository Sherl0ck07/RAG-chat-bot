"""
Quick query test - Single query to test the API
"""

import requests
import sys

def query_api(question: str):
    """Send a query to the API and print the response"""
    url = "http://localhost:8000/chat"
    
    try:
        response = requests.post(
            url,
            json={"query": question},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        print("\n" + "="*70)
        print(f"QUERY: {question}")
        print("="*70)
        print(f"\nANSWER:\n{data['answer']}\n")
        print(f"CONFIDENCE: {data.get('confidence', 'N/A')}")
        print(f"\nSOURCES ({len(data.get('sources', []))}):")
        
        for i, src in enumerate(data.get('sources', []), 1):
            print(f"  [{i}] {src.get('section', 'Unknown')} - Score: {src.get('similarity_score', 0):.4f}")
        
        print("="*70 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API. Is the server running?")
        print("   Start server with: python app/main.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is Accelerate?"
    
    query_api(question)