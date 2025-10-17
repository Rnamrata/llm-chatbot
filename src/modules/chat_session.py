import uuid
from datetime import datetime, timedelta


class ChatSession:
    """Manages individual chat sessions and conversation history"""
    
    def __init__(self, llm_manager, vector_store):
        """
        Initialize Chat Session Manager
        
        Args:
            llm_manager: LLMManager instance
            vector_store: VectorStoreAndEmbedding instance
        """
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.sessions = {}
    
    def create_session(self, session_id=None, k=5):
        """
        Create a new chat session
        
        Args:
            session_id: Optional session ID, generates one if not provided
            k: Number of documents to retrieve
        
        Returns:
            str: Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            return session_id
        
        # Create retriever from vector store
        retriever = self.vector_store.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Create conversational chain using LLM manager
        chain, memory = self.llm_manager.create_conversational_chain(retriever, k)
        
        # Store session data
        self.sessions[session_id] = {
            'chain': chain,
            'memory': memory,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'message_count': 0
        }
        
        print(f"✅ Created new session: {session_id}")
        return session_id
    
    def query(self, question, session_id, k=5):
        """
        Query with conversation history
        
        Args:
            question: User's question
            session_id: Session identifier
            k: Number of documents to retrieve
        
        Returns:
            dict: Response with answer, sources, and metadata
        """
        try:
            # Create session if it doesn't exist
            if session_id not in self.sessions:
                self.create_session(session_id, k)
            
            session = self.sessions[session_id]
            
            # Run the chain
            result = session['chain']({"question": question})
            
            # Update session metadata
            session['last_activity'] = datetime.now()
            session['message_count'] += 1
            
            # Format sources using LLM manager
            sources = self.llm_manager.format_sources(
                result.get('source_documents', [])
            )
            
            return {
                'success': True,
                'answer': result['answer'].strip(),
                'sources': sources,
                'num_sources': len(result.get('source_documents', [])),
                'session_id': session_id,
                'message_count': session['message_count']
            }
            
        except Exception as e:
            print(f"Error during query: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': 'An error occurred while processing your query.',
                'session_id': session_id
            }
    
    def get_session_info(self, session_id):
        """
        Get information about a session
        
        Args:
            session_id: Session identifier
        
        Returns:
            dict: Session information
        """
        if session_id not in self.sessions:
            return {
                'exists': False,
                'message': 'Session not found'
            }
        
        session = self.sessions[session_id]
        return {
            'exists': True,
            'session_id': session_id,
            'created_at': session['created_at'].isoformat(),
            'last_activity': session['last_activity'].isoformat(),
            'message_count': session['message_count']
        }
    
    def get_history(self, session_id):
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
        
        Returns:
            dict: Conversation history
        """
        if session_id not in self.sessions:
            return {
                'history': [],
                'length': 0,
                'message': 'Session not found'
            }
        
        session = self.sessions[session_id]
        memory = session['memory']
        messages = memory.chat_memory.messages
        
        # Format history as Q&A pairs
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append({
                    'question': messages[i].content,
                    'answer': messages[i + 1].content,
                    'timestamp': session['created_at'].isoformat()
                })
        
        return {
            'history': history,
            'length': len(history),
            'session_id': session_id,
            'message_count': session['message_count']
        }
    
    def clear_history(self, session_id):
        """
        Clear conversation history for a session
        
        Args:
            session_id: Session identifier
        
        Returns:
            dict: Result of clearing operation
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"🗑️  Cleared session: {session_id}")
            return {
                'success': True,
                'message': 'Conversation history cleared',
                'session_id': session_id
            }
        
        return {
            'success': False,
            'message': 'Session not found',
            'session_id': session_id
        }
    
    def clear_all_sessions(self):
        """
        Clear all sessions
        
        Returns:
            dict: Result of clearing operation
        """
        count = len(self.sessions)
        self.sessions = {}
        print(f"🗑️  Cleared {count} sessions")
        return {
            'success': True,
            'message': f'Cleared {count} sessions',
            'count': count
        }
    
    def list_sessions(self):
        """
        List all active sessions
        
        Returns:
            dict: List of session information
        """
        sessions_info = []
        for session_id, session in self.sessions.items():
            sessions_info.append({
                'session_id': session_id,
                'created_at': session['created_at'].isoformat(),
                'last_activity': session['last_activity'].isoformat(),
                'message_count': session['message_count']
            })
        
        return {
            'sessions': sessions_info,
            'total_sessions': len(sessions_info)
        }
    
    def cleanup_inactive_sessions(self, inactive_hours=24):
        """
        Remove sessions inactive for specified hours
        
        Args:
            inactive_hours: Hours of inactivity before cleanup
        
        Returns:
            dict: Cleanup results
        """
        
        now = datetime.now()
        cutoff = now - timedelta(hours=inactive_hours)
        
        inactive_sessions = [
            sid for sid, session in self.sessions.items()
            if session['last_activity'] < cutoff
        ]
        
        for session_id in inactive_sessions:
            del self.sessions[session_id]
        
        print(f"🧹 Cleaned up {len(inactive_sessions)} inactive sessions")
        
        return {
            'success': True,
            'cleaned': len(inactive_sessions),
            'remaining': len(self.sessions),
            'message': f'Cleaned up {len(inactive_sessions)} inactive sessions'
        }