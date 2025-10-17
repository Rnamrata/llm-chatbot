# test_system.py
import requests
import json
import time
import sys

BASE_URL = 'http://localhost:5001'

def print_separator(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f'{BASE_URL}/health', timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("   Make sure the Flask server is running:")
        print("   python main.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ Server request timed out!")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def safe_request(method, url, **kwargs):
    """Make a safe HTTP request with error handling"""
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=10, **kwargs)
        elif method.upper() == 'POST':
            response = requests.post(url, timeout=30, **kwargs)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, timeout=10, **kwargs)
        else:
            print(f"❌ Unsupported method: {method}")
            return None
        
        # Check if response is JSON
        try:
            return response.json()
        except:
            print(f"❌ Server returned non-JSON response:")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the server running?")
        return None
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_health():
    print_separator("Health Check")
    result = safe_request('GET', f'{BASE_URL}/health')
    if result:
        print(json.dumps(result, indent=2))
        return True
    return False

def test_file_upload():
    print_separator("Test 1: File Upload")
    
    import os
    if not os.path.exists('test.pdf'):
        print("⚠️  test.pdf not found. Skipping this test.")
        print("   Create a test.pdf file or use test_web_upload instead.")
        return False
    
    try:
        with open('test.pdf', 'rb') as f:
            response = requests.post(
                f'{BASE_URL}/upload/file',
                files={'file': f},
                timeout=30
            )
        result = response.json()
        print(json.dumps(result, indent=2))
        return result.get('success', False)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_web_upload():
    print_separator("Test 2: Web Page Upload")
    result = safe_request(
        'POST',
        f'{BASE_URL}/upload/web',
        json={'url': 'https://en.wikipedia.org/wiki/Machine_learning'}
    )
    if result:
        print(json.dumps(result, indent=2))
        return result.get('success', False)
    return False

def test_stats():
    print_separator("Database Statistics")
    result = safe_request('GET', f'{BASE_URL}/stats')
    if result:
        print(json.dumps(result, indent=2))
        return True
    return False

def test_chat_session():
    print_separator("Test 3: Chat Session")
    
    # Create new session
    print("Creating new session...")
    result = safe_request('POST', f'{BASE_URL}/chat/new')
    if not result:
        return False
    
    session_id = result['session_id']
    print(f"✅ Session created: {session_id}\n")
    
    # Ask questions
    questions = [
        "What is Machine learning?",
        "Can you explain Python programming language?",
        "Give me a summary"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*60}")
        print(f"Q{i}: {question}")
        
        result = safe_request(
            'POST',
            f'{BASE_URL}/chat',
            json={'question': question, 'session_id': session_id}
        )
        
        if result and result.get('success'):
            answer = result.get('answer', '')
            print(f"A{i}: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"📚 Sources: {result.get('num_sources', 0)}")
            print(f"💬 Messages: {result.get('message_count', 0)}")
        else:
            print(f"❌ Chat failed: {result.get('error') if result else 'No response'}")
        
        time.sleep(1)
    
    # Get session info
    print(f"\n{'─'*60}")
    print("📊 Session Information:")
    result = safe_request('GET', f'{BASE_URL}/chat/session/{session_id}')
    if result:
        print(json.dumps(result, indent=2))
    
    # Get history
    print(f"\n{'─'*60}")
    print("📜 Conversation History:")
    result = safe_request('GET', f'{BASE_URL}/chat/history/{session_id}')
    if result:
        print(f"Total exchanges: {result.get('length', 0)}")
    
    return True

def test_cleanup():
    print_separator("Test 4: Session Cleanup")
    result = safe_request(
        'POST',
        f'{BASE_URL}/chat/cleanup',
        json={'inactive_hours': 1}
    )
    if result:
        print(json.dumps(result, indent=2))
        return True
    return False

def main():
    print("\n🧪 Testing RAG System\n")
    
    # First check if server is running
    print_separator("Server Connection Check")
    if not check_server():
        print("\n" + "="*60)
        print("❌ Tests aborted - server not available")
        print("="*60)
        print("\nTo start the server, run:")
        print("   python main.py")
        sys.exit(1)
    
    # Run tests
    results = {
        'health': test_health(),
        'stats': test_stats(),
        # 'file_upload': test_file_upload(),  # Uncomment if you have test.pdf
        'web_upload': test_web_upload(),
        'chat': test_chat_session(),
        # 'cleanup': test_cleanup(),  # Uncomment to test cleanup
    }
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*60 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrupted by user")
        sys.exit(0)