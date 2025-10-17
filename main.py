from flask import Flask, request, jsonify
from flask_cors import CORS
import src.modules.file_manager as file_manager_module
import src.modules.document_processor as document_processor_module
import src.modules.vector_store_and_embedding as vector_store_module
import src.modules.llm_manager as llm_manager_module
import src.modules.chat_session as chat_session_module
import uuid
import os

app = Flask(__name__)
CORS(app)

document_processor = document_processor_module.DocumentProcessor()
vector_store = vector_store_module.VectorStoreAndEmbedding()
file_manager = file_manager_module.FileManager(document_processor, vector_store)
llm_manager = llm_manager_module.LLMManager(model_name="llama3.2", temperature=0.7)
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
    result = file_manager.uploadFile()
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
    result = file_manager.uploadMediaFile()
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
    result = file_manager.webFileUpload()
    status_code = 200 if result.get('success') else 400
    return jsonify(result), status_code

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
        
        return jsonify({
            'total_chunks': count,
            'total_sessions': sessions['total_sessions'],
            'status': 'ready' if count > 0 else 'empty',
            'message': f'Vector database contains {count} chunks'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'RAG System',
        'version': '1.0',
        'llm_model': llm_manager.model_name
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
    print("🌐 Server starting on http://0.0.0.0:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
