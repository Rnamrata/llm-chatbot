from flask import Flask, request, jsonify
from flask_cors import CORS
import src.modules.file_manager as file_manager_module
import src.modules.document_processor as document_processor_module
import src.modules.vector_store_and_embedding as vector_store_module
import src.modules.llm_manager as llm_manager_module
import src.modules.chat_session as chat_session_module
import src.modules.review_client as review_client_module
from src import config
import uuid
import requests 

app = Flask(__name__)
CORS(app, origins=config.CORS_ORIGINS)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_BYTES

document_processor = document_processor_module.DocumentProcessor()
vector_store = vector_store_module.VectorStoreAndEmbedding()
review_client = review_client_module.ReviewClient()
file_manager = file_manager_module.FileManager(document_processor, vector_store, review_client)
llm_manager = llm_manager_module.LLMManager(model_name=None, temperature=None)
chat_manager = chat_session_module.ChatSession(llm_manager, vector_store)

# ==================== UPLOAD ENDPOINTS ====================

@app.route('/upload/file', methods=['POST'])
def upload_file():
    """
    Upload a file from device
    Automatically processes, chunks, embeds, and stores
    
    Form data:
        - file: The file to upload (PDF, TXT, MD)
    """
    file = request.files.get('file')
    if not file:
       return jsonify({'error': 'No file provided', 'success': False}), 400
    result = file_manager.uploadFile(file)

    status_code = 200 if result.get('success') else 400
    return jsonify(result), status_code


@app.route('/upload/youtube', methods=['POST'])
def upload_youtube():
    """
    Upload YouTube video
    Automatically downloads, transcribes, chunks, embeds, and stores
    
    JSON or Form data:
        - url: YouTube video URL
    """
    url = request.form.get('url') or (request.get_json(silent=True) or {}).get('url')
    result = file_manager.uploadMediaFile(url)
    
    status_code = 200 if result.get('success') else 400
    return jsonify(result), status_code


@app.route('/upload/web', methods=['POST'])
def upload_web():
    """
    Upload web page content
    Automatically scrapes, chunks, embeds, and stores
    
    JSON or Form data:
        - url: Web page URL
    """
    url = request.form.get('url') or (request.get_json(silent=True) or {}).get('url')
    result = file_manager.webFileUpload(url)

    status_code = 200 if result.get('success') else 400
    return jsonify(result), status_code

@app.route('/upload/code', methods=['POST'])
def upload_code():
    """
    Upload code file for review
    Automatically validates, chunks, embeds, and stores
    
    JSON or Form data:
        - file: The code file to upload (required)
        - session_id: Session ID for code review (required)
    """
    file = request.files.get('file')
    session_id = request.form.get('session_id') or (request.get_json(silent=True) or {}).get('session_id')
    if not file:
        return jsonify({'error': 'No file provided', 'success': False}), 400
    if not session_id:
        return jsonify({'error': 'No session_id provided', 'success': False}), 400
    result = file_manager.uploadCodeForReview(file, session_id)

    status_code = 200 if result.get('success') else 400
    return jsonify(result), status_code

@app.route('/review', methods=['POST'])
def review_code():
    """
    Upload a code file for automated review, attached to a chat session
    Automatically: Review (Node) → Store → Attach to session

    Form data:
        - file: The code file to review (required)
        - session_id: Existing session to attach the review to (optional,
                       a new session is created if omitted)
    """
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided', 'success': False}), 400

    session_id = request.form.get('session_id') or (request.get_json(silent=True) or {}).get('session_id')
    if not session_id:
        session_id = chat_manager.create_session()

    result = file_manager.uploadCodeForReview(file, session_id)
    if not result.get('success'):
        return jsonify(result), 400

    summary = llm_manager.summarize_review(result['findings'])
    chat_manager.attach_review(session_id, result['review_id'], summary)

    response = {
        'success': True,
        'session_id': session_id,
        'review_id': result['review_id'],
        'summary': summary,
        'findings': result['findings'],
        'code': result['code'],
        'language': result['language']
    }

    if result.get('warning'):
        response['warning'] = result['warning']

    return jsonify(response)

@app.route('/review/<review_id>', methods=['GET'])
def get_review(review_id):
    """
    Get a previously stored review's code and findings by review_id
    So Angular can reload a review after a page refresh
    """
    try:
        result = vector_store.vectorstore.get(
            where={'review_id': review_id},
            include=['documents', 'metadatas']
        )

        if not result['ids']:
            return jsonify({'error': 'Review not found', 'success': False}), 404

        findings = []
        code = None
        filename = None
        language = None

        for doc_text, metadata in zip(result['documents'], result['metadatas']):
            doc_type = metadata.get('type')
            if doc_type == 'code_review_source':
                code = doc_text
                filename = metadata.get('source')
                language = metadata.get('language')
            elif doc_type == 'review_finding':
                findings.append({
                    'file': metadata.get('file'),
                    'line': metadata.get('line'),
                    'severity': metadata.get('severity'),
                    'message': metadata.get('message')
                })

        return jsonify({
            'success': True,
            'review_id': review_id,
            'filename': filename,
            'language': language,
            'code': code,
            'findings': findings
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# ==================== CHAT ENDPOINTS ====================

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat with your documents
    
    JSON body:
        - question: User's question (required)
        - session_id: Session ID for conversation continuity (optional)
        - k: Number of relevant chunks to retrieve (optional, default: 5)
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question']
        session_id = data.get('session_id', str(uuid.uuid4()))
        k = data.get('k', 5)
        
        result = chat_manager.query(question, session_id, k=k)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat/new', methods=['POST'])
def new_chat():
    """Start a new chat session"""
    session_id = str(uuid.uuid4())
    chat_manager.create_session(session_id)
    return jsonify({
        'session_id': session_id,
        'message': 'New chat session created'
    })


@app.route('/chat/session/<session_id>', methods=['GET'])
def get_session_info(session_id):
    """Get information about a specific session"""
    info = chat_manager.get_session_info(session_id)
    return jsonify(info)


@app.route('/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get chat history for a specific session"""
    history = chat_manager.get_history(session_id)
    return jsonify(history)


@app.route('/chat/clear/<session_id>', methods=['DELETE'])
def clear_chat_history(session_id):
    """Clear chat history for a specific session"""
    result = chat_manager.clear_history(session_id)
    return jsonify(result)


@app.route('/chat/sessions', methods=['GET'])
def list_sessions():
    """List all active chat sessions"""
    result = chat_manager.list_sessions()
    return jsonify(result)


@app.route('/chat/cleanup', methods=['POST'])
def cleanup_sessions():
    """Clean up inactive sessions"""
    data = request.get_json() or {}
    inactive_hours = data.get('inactive_hours', 24)
    result = chat_manager.cleanup_inactive_sessions(inactive_hours)
    return jsonify(result)


# ==================== UTILITY ENDPOINTS ====================

@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics about the vector database"""
    try:
        count = vector_store.vectorstore._collection.count()
        sessions = chat_manager.list_sessions()

        reviews = vector_store.vectorstore.get(
            where={'type': 'code_review_source'},
            include=[]
        )
        
        return jsonify({
            'total_chunks': count,
            'total_sessions': sessions['total_sessions'],
            'total_reviews': len(reviews['ids']),
            'status': 'ready' if count > 0 else 'empty',
            'message': f'Vector database contains {count} chunks'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        ollama_response = requests.get(config.OLLAMA_BASE_URL, timeout=2)
        ollama_status = 'healthy' if ollama_response.status_code == 200 else f'unhealthy ({ollama_response.status_code})'
    except requests.exceptions.RequestException as e:
        ollama_status = f'unreachable: {e}'

    review_status = review_client.health()
    return jsonify({
        'status': 'healthy',
        'service': 'RAG System',
        'version': '1.0',
        'llm_model': llm_manager.model_name,
        'ollama': ollama_status,
        'review_service': review_status        
    })


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== RUN APP ====================
if __name__ == "__main__":
    print(f"🌐 Server starting on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
