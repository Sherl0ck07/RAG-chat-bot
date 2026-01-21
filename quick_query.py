"""
Quick query test - Single query to test the API
"""

import requests
import sys

def query_api(question: str):
    """Send a query to the API and print the response"""
    print(f"[QUICK_QUERY] Sending query: {question}")
    url = "http://localhost:8000/chat"
    print(f"[QUICK_QUERY] API endpoint: {url}")
    
    try:
        print(f"[QUICK_QUERY] Creating request...")
        response = requests.post(
            url,
            json={"query": question},
            timeout=30
        )
        print(f"[QUICK_QUERY] Response status: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        print(f"[QUICK_QUERY] Response received successfully")
        print(f"[QUICK_QUERY] Answer length: {len(data.get('answer', ''))} chars")
        print(f"[QUICK_QUERY] Confidence: {data.get('confidence', 'N/A')}")
        print(f"[QUICK_QUERY] Number of sources: {len(data.get('sources', []))}")
        
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
        print("[QUICK_QUERY] ERROR: Cannot connect to API. Is the server running?")
        print("[QUICK_QUERY] Start server with: python app/main.py")
    except Exception as e:
        print(f"[QUICK_QUERY] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("[QUICK_QUERY] =====================================")
    print("[QUICK_QUERY] Quick Query Test Script")
    print("[QUICK_QUERY] =====================================")
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"[QUICK_QUERY] Using command-line question")
    else:
        question = "What is Accelerate?"
        print(f"[QUICK_QUERY] Using default question")
    
    query_api(question)