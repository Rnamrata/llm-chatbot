# Import UUID library for generating unique session identifiers
import uuid
# Import datetime classes for tracking session creation and activity times
from datetime import datetime, timedelta


class ChatSession:
    """
    Chat Session Manager
    ====================
    Manages individual chat sessions with conversation history and RAG (Retrieval-Augmented Generation).

    This class is responsible for:
    - Creating and managing multiple independent chat sessions
    - Maintaining conversation history for each session
    - Integrating with the vector store for context retrieval
    - Managing session lifecycle (creation, queries, cleanup)

    Each session has its own:
    - Conversational chain (LLM + retriever)
    - Memory (conversation history)
    - Metadata (creation time, last activity, message count)

    This allows multiple users or conversations to exist simultaneously without interference.
    """

    def __init__(self, llm_manager, vector_store):
        """
        Initialize the Chat Session Manager with required dependencies.

        The manager doesn't create any sessions initially - sessions are created
        on-demand when a user starts chatting.

        Args:
            llm_manager (LLMManager): Instance that handles LLM initialization and queries
            vector_store (VectorStoreAndEmbedding): Instance that manages embeddings and retrieval

        Instance Variables:
            self.llm_manager: Reference to LLM manager for creating conversational chains
            self.vector_store: Reference to vector store for document retrieval
            self.sessions: Dictionary mapping session_id -> session_data
                          Structure: {
                              'session_id': {
                                  'chain': ConversationalRetrievalChain object,
                                  'memory': ConversationBufferMemory object,
                                  'created_at': datetime,
                                  'last_activity': datetime,
                                  'message_count': int
                              }
                          }
        """
        self.llm_manager = llm_manager           # For creating LLM chains
        self.vector_store = vector_store         # For retrieving relevant documents
        self.sessions = {}                       # Empty dict - sessions created on demand
    
    def create_session(self, session_id=None, k=5):
        """
        Create a new chat session with its own conversation history and RAG chain.

        This method sets up everything needed for a conversational RAG session:
        1. Generates a unique session ID (if not provided)
        2. Creates a retriever that will fetch relevant documents from vector store
        3. Creates a conversational chain that combines the LLM with the retriever
        4. Initializes memory to track conversation history
        5. Stores all session data for future queries

        The session remains active until explicitly cleared or cleaned up by inactivity timeout.

        Args:
            session_id (str, optional): Custom session ID. If None, generates a UUID.
            k (int): Number of relevant documents to retrieve for each query (default: 5)
                    Higher k = more context but longer prompts and slower responses

        Returns:
            str: The session ID (either provided or generated)

        Note: If session_id already exists, returns existing ID without creating new session
        """
        # Generate a unique session ID if one wasn't provided
        # UUID4 generates a random 128-bit identifier (e.g., "550e8400-e29b-41d4-a716-446655440000")
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Check if this session already exists
        if session_id in self.sessions:
            # Session exists, just return the ID without creating a duplicate
            return session_id

        # Create a retriever from the vector store
        # The retriever will search for the k most similar documents to each query
        # search_kwargs={'k': k} tells it how many documents to retrieve
        retriever = self.vector_store.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # Create a conversational chain using the LLM manager
        # This combines:
        # - The LLM (for generating responses)
        # - The retriever (for finding relevant documents)
        # - Memory (for maintaining conversation history)
        # Returns: (chain, memory) tuple
        chain, memory = self.llm_manager.create_conversational_chain(retriever, k)

        # Store all session data in the sessions dictionary
        self.sessions[session_id] = {
            'chain': chain,                      # The conversational RAG chain
            'memory': memory,                    # Conversation history buffer
            'created_at': datetime.now(),        # When session was created
            'last_activity': datetime.now(),     # Last time session was used
            'message_count': 0                   # Number of messages exchanged
        }

        # Log session creation
        print(f"✅ Created new session: {session_id}")
        return session_id
    
    def query(self, question, session_id, k=5):
        """
        Process a user query with RAG (Retrieval-Augmented Generation) and conversation history.

        This is the main method for answering user questions. It:
        1. Creates a session if it doesn't exist
        2. Retrieves relevant documents from the vector store
        3. Passes question + documents + conversation history to the LLM
        4. Gets an answer that's grounded in the retrieved documents
        5. Updates session metadata and returns formatted response

        The conversational aspect means the LLM can reference previous messages,
        allowing for natural follow-up questions like "tell me more about that".

        Args:
            question (str): The user's question or message
            session_id (str): Unique identifier for this conversation session
            k (int): Number of relevant documents to retrieve (default: 5)

        Returns:
            dict: Response dictionary containing:
                - success (bool): Whether the query succeeded
                - answer (str): The LLM's response
                - sources (list): Formatted list of source documents used
                - num_sources (int): Number of source documents retrieved
                - session_id (str): The session ID
                - message_count (int): Total messages in this session

                On error:
                - success (bool): False
                - error (str): Error message
                - answer (str): User-friendly error message
                - session_id (str): The session ID
        """
        try:
            # Create session if it doesn't exist (auto-creation for convenience)
            if session_id not in self.sessions:
                self.create_session(session_id, k)

            # Get the session data (chain, memory, metadata)
            session = self.sessions[session_id]

            # Run the conversational RAG chain
            # This does several things behind the scenes:
            # 1. Uses the retriever to find k relevant documents
            # 2. Loads conversation history from memory
            # 3. Passes everything to the LLM
            # 4. Stores the Q&A pair in memory for future reference
            # Input: {"question": "What is...?"}
            # Output: {"answer": "...", "source_documents": [...]}
            result = session['chain']({"question": question})

            # Update session metadata to track activity
            session['last_activity'] = datetime.now()  # Update last used time (for cleanup)
            session['message_count'] += 1              # Increment message counter

            # Format the source documents into a readable structure
            # Truncates content, extracts metadata, etc.
            sources = self.llm_manager.format_sources(
                result.get('source_documents', [])
            )

            # Return structured response
            return {
                'success': True,
                'answer': result['answer'].strip(),                    # LLM's answer
                'sources': sources,                                     # Where the info came from
                'num_sources': len(result.get('source_documents', [])),# How many sources used
                'session_id': session_id,                              # Echo back session ID
                'message_count': session['message_count']              # Total Q&A pairs in session
            }

        except Exception as e:
            # Catch any errors (LLM timeout, retrieval failure, etc.)
            print(f"Error during query: {e}")
            return {
                'success': False,
                'error': str(e),                                       # Technical error message
                'answer': 'An error occurred while processing your query.',  # User-friendly message
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