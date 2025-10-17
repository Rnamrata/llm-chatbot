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


@app.route('/upload/code', methods=['POST'])
def upload_code():
    """
    Upload code file for review
    Provides complexity analysis and code-specific chunking

    Form data:
        - file: Code file to upload (.py, .js, .ts, .java, .go, etc.)
    """
    result = file_manager.uploadCodeForReview()
    status_code = 200 if result.get('success') else 400
    return jsonify(result), status_code


# ==================== CODE REVIEW ENDPOINTS ====================

@app.route('/review/quick', methods=['POST'])
def quick_review():
    """
    Quick code review focusing on critical issues

    Form data:
        - file: Code file to review
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        filename = file.filename

        # Read code content
        code_content = file.read().decode('utf-8')

        # Detect language
        from src.modules.code_parser import CodeParser
        parser = CodeParser()
        language = parser.detect_language(filename)

        if language == 'unknown':
            return jsonify({'error': 'Unsupported file type'}), 400

        # Perform quick review
        review = llm_manager.review_code_direct(
            code=code_content,
            language=language,
            source=filename,
            review_type="quick"
        )

        return jsonify({
            'success': True,
            'filename': filename,
            'language': language,
            'review_type': 'quick',
            'review': review
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/review/comprehensive', methods=['POST'])
def comprehensive_review():
    """
    Comprehensive code review with detailed analysis

    Form data or JSON:
        - file: Code file to review (Form)
        OR
        - code: Code content as string (JSON)
        - filename: Name of the file (JSON)
        - question: Optional specific focus (JSON)
    """
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            code_content = file.read().decode('utf-8')
            question = request.form.get('question', 'Please provide a comprehensive review.')
        # Handle JSON request
        elif request.is_json:
            data = request.get_json()
            code_content = data.get('code')
            filename = data.get('filename', 'code_snippet')
            question = data.get('question', 'Please provide a comprehensive review.')

            if not code_content:
                return jsonify({'error': 'No code provided'}), 400
        else:
            return jsonify({'error': 'No code provided'}), 400

        # Detect language
        from src.modules.code_parser import CodeParser
        parser = CodeParser()
        language = parser.detect_language(filename)

        # Get related code from vector store for context
        context_docs = vector_store.search(code_content[:500], k=3)

        # Perform comprehensive review
        review = llm_manager.review_code_with_context(
            code=code_content,
            language=language,
            source=filename,
            context_documents=context_docs,
            question=question,
            review_type="comprehensive"
        )

        return jsonify({
            'success': True,
            'filename': filename,
            'language': language,
            'review_type': 'comprehensive',
            'review': review,
            'context_used': len(context_docs)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/review/security', methods=['POST'])
def security_review():
    """
    Security-focused code review

    Form data:
        - file: Code file to review
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        filename = file.filename
        code_content = file.read().decode('utf-8')

        from src.modules.code_parser import CodeParser
        parser = CodeParser()
        language = parser.detect_language(filename)

        # Get context
        context_docs = vector_store.search(code_content[:500], k=3)

        # Perform security review
        review = llm_manager.review_code_with_context(
            code=code_content,
            language=language,
            source=filename,
            context_documents=context_docs,
            question="Analyze for security vulnerabilities.",
            review_type="security"
        )

        return jsonify({
            'success': True,
            'filename': filename,
            'language': language,
            'review_type': 'security',
            'review': review
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/review/performance', methods=['POST'])
def performance_review():
    """
    Performance-focused code review

    Form data:
        - file: Code file to review
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        filename = file.filename
        code_content = file.read().decode('utf-8')

        from src.modules.code_parser import CodeParser
        parser = CodeParser()
        language = parser.detect_language(filename)

        # Get context
        context_docs = vector_store.search(code_content[:500], k=3)

        # Perform performance review
        review = llm_manager.review_code_with_context(
            code=code_content,
            language=language,
            source=filename,
            context_documents=context_docs,
            question="Analyze for performance issues and optimization opportunities.",
            review_type="performance"
        )

        return jsonify({
            'success': True,
            'filename': filename,
            'language': language,
            'review_type': 'performance',
            'review': review
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/review/explain', methods=['POST'])
def explain_code():
    """
    Explain what code does

    Form data or JSON:
        - file: Code file (Form)
        OR
        - code: Code content (JSON)
        - filename: File name (JSON)
        - question: Specific question about code (optional)
    """
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            code_content = file.read().decode('utf-8')
            question = request.form.get('question', 'What does this code do?')
        # Handle JSON request
        elif request.is_json:
            data = request.get_json()
            code_content = data.get('code')
            filename = data.get('filename', 'code_snippet')
            question = data.get('question', 'What does this code do?')

            if not code_content:
                return jsonify({'error': 'No code provided'}), 400
        else:
            return jsonify({'error': 'No code provided'}), 400

        from src.modules.code_parser import CodeParser
        parser = CodeParser()
        language = parser.detect_language(filename)

        # Get context
        context_docs = vector_store.search(code_content[:500], k=3)

        # Explain code
        explanation = llm_manager.explain_code(
            code=code_content,
            language=language,
            source=filename,
            context_documents=context_docs,
            question=question
        )

        return jsonify({
            'success': True,
            'filename': filename,
            'language': language,
            'explanation': explanation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/review/bugs', methods=['POST'])
def detect_bugs():
    """
    Detect potential bugs in code

    Form data or JSON:
        - file: Code file (Form)
        OR
        - code: Code content (JSON)
        - filename: File name (JSON)
        - issue: Reported issue description (optional)
    """
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            code_content = file.read().decode('utf-8')
            issue = request.form.get('issue', '')
        # Handle JSON request
        elif request.is_json:
            data = request.get_json()
            code_content = data.get('code')
            filename = data.get('filename', 'code_snippet')
            issue = data.get('issue', '')

            if not code_content:
                return jsonify({'error': 'No code provided'}), 400
        else:
            return jsonify({'error': 'No code provided'}), 400

        from src.modules.code_parser import CodeParser
        parser = CodeParser()
        language = parser.detect_language(filename)

        # Get context
        context_docs = vector_store.search(code_content[:500], k=3)

        # Detect bugs
        analysis = llm_manager.detect_bugs(
            code=code_content,
            language=language,
            source=filename,
            reported_issue=issue,
            context_documents=context_docs
        )

        return jsonify({
            'success': True,
            'filename': filename,
            'language': language,
            'bug_analysis': analysis
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/review/improve', methods=['POST'])
def suggest_improvements():
    """
    Suggest code improvements

    Form data or JSON:
        - file: Code file (Form)
        OR
        - code: Code content (JSON)
        - filename: File name (JSON)
        - goal: Improvement goal (optional)
    """
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            code_content = file.read().decode('utf-8')
            goal = request.form.get('goal', '')
        # Handle JSON request
        elif request.is_json:
            data = request.get_json()
            code_content = data.get('code')
            filename = data.get('filename', 'code_snippet')
            goal = data.get('goal', '')

            if not code_content:
                return jsonify({'error': 'No code provided'}), 400
        else:
            return jsonify({'error': 'No code provided'}), 400

        from src.modules.code_parser import CodeParser
        parser = CodeParser()
        language = parser.detect_language(filename)

        # Get context
        context_docs = vector_store.search(code_content[:500], k=3)

        # Suggest improvements
        suggestions = llm_manager.suggest_improvements(
            code=code_content,
            language=language,
            source=filename,
            goal=goal,
            context_documents=context_docs
        )

        return jsonify({
            'success': True,
            'filename': filename,
            'language': language,
            'suggestions': suggestions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
