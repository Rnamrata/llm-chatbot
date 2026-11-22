import uuid
from datetime import datetime, timedelta

# Longest title made from a first question
TITLE_LENGTH = 60


def _shorten(text, limit):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def _parse_time(value, default):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return default


class ChatSession:
    """
    Manages chat sessions: their conversation, metadata (title, type, pinned) and,
    when a store is given, saving them so they survive a restart
    """

    def __init__(self, llm_manager, vector_store, store=None):
        """
        Initialize Chat Session Manager

        Args:
            llm_manager: LLMManager instance
            vector_store: VectorStoreAndEmbedding instance
            store: SessionStore to save sessions in (optional; in memory only without it)
        """
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.store = store
        self.sessions = {}

        if store is not None:
            for session_id, record in store.load().items():
                self.sessions[session_id] = self._from_record(session_id, record)
            if self.sessions:
                print(f"📂 Loaded {len(self.sessions)} saved sessions")

    # ---------- sessions ----------

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

        now = datetime.now()
        self.sessions[session_id] = {
            'chain': None,  # built on first use, see _chain()
            'memory': None,
            'mode': 'docs',
            'review_id': None,
            'filename': None,
            'language': None,
            'title': None,
            'pinned': False,
            'k': k,
            'messages': [],
            'created_at': now,
            'last_activity': now,
        }

        print(f"✅ Created new session: {session_id}")
        return session_id

    def attach_review(self, session_id, review_id, summary, filename=None, language=None):
        """
        Turn a session into a code-review chat: retrieval is scoped to one review's
        chunks, and the conversation opens with the review summary

        Args:
            session_id: Session to attach the review to (created if missing)
            review_id: The review_id to scope retrieval to (set by uploadCodeForReview)
            summary: Opening summary text, e.g. from review_summary.build_headline
            filename: The reviewed file, for the title (optional)
            language: The file's language, e.g. 'python' (optional)

        Returns:
            dict: {'success': True, 'session_id': ..., 'review_id': ...}
        """
        if session_id not in self.sessions:
            self.create_session(session_id)

        session = self.sessions[session_id]
        now = datetime.now()
        session.update({
            'chain': None,
            'memory': None,
            'mode': 'code_review',
            'review_id': review_id,
            'filename': filename or session.get('filename'),
            'language': language or session.get('language'),
            'messages': [{'role': 'assistant', 'content': summary, 'timestamp': now, 'sources': []}],
            'last_activity': now,
        })
        if not session['title'] and session['filename']:
            session['title'] = f"Review: {session['filename']}"

        self._save()
        print(f"🔗 Attached review {review_id} to session: {session_id}")

        return {
            'success': True,
            'session_id': session_id,
            'review_id': review_id
        }

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
            chain = self._chain(session)

            # If a different k comes in, update the retriever in place —
            # this keeps the existing chain/memory, so history isn't lost
            if k != session['k']:
                chain.retriever = self._retriever(session, k)
                session['k'] = k

            # Run the chain
            result = chain({"question": question})
            answer = result['answer'].strip()

            # Format sources using LLM manager
            sources = self.llm_manager.format_sources(
                result.get('source_documents', [])
            )

            now = datetime.now()
            session['messages'].append({'role': 'user', 'content': question, 'timestamp': now})
            session['messages'].append({'role': 'assistant', 'content': answer, 'timestamp': now, 'sources': sources})
            session['last_activity'] = now
            if not session['title']:
                session['title'] = _shorten(question, TITLE_LENGTH)
            self._save()

            response = {
                'success': True,
                'answer': answer,
                'sources': sources,
                'num_sources': len(result.get('source_documents', [])),
                'session_id': session_id,
                'message_count': self._question_count(session)
            }

            if session['review_id']:
                response['review_id'] = session['review_id']

            return response

        except Exception as e:
            print(f"Error during query: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': 'An error occurred while processing your query.',
                'session_id': session_id
            }

    def update_session(self, session_id, pinned=None, title=None):
        """
        Pin, unpin or rename a session

        Returns:
            dict: The session as list_sessions shows it, or None if there is no such session
        """
        session = self.sessions.get(session_id)
        if session is None:
            return None
        if pinned is not None:
            session['pinned'] = bool(pinned)
        if title is not None:
            session['title'] = _shorten(title, 120) or session['title']
        self._save()
        return self._summary(session_id, session)

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

        return {'exists': True, **self._summary(session_id, self.sessions[session_id])}

    def get_history(self, session_id):
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier

        Returns:
            dict: Conversation history: question and answer pairs, each answer with
                  its sources. A standalone answer (the review summary) has no question.
        """
        if session_id not in self.sessions:
            return {
                'history': [],
                'length': 0,
                'message': 'Session not found'
            }

        session = self.sessions[session_id]
        messages = session['messages']

        history = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg['role'] == 'user' and i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                answer = messages[i + 1]
                history.append({
                    'question': msg['content'],
                    'answer': answer['content'],
                    'sources': answer.get('sources', []),
                    'timestamp': answer['timestamp'].isoformat()
                })
                i += 2
            else:
                history.append({
                    'question': None,
                    'answer': msg['content'],
                    'sources': msg.get('sources', []),
                    'timestamp': msg['timestamp'].isoformat()
                })
                i += 1

        return {
            'history': history,
            'length': len(history),
            'session_id': session_id,
            'message_count': self._question_count(session),
            'title': session['title'],
            'type': session['mode'],
            'review_id': session['review_id'],
        }

    def clear_history(self, session_id):
        """
        Delete a session. For a review's session, the stored review goes too.

        Args:
            session_id: Session identifier

        Returns:
            dict: Result of clearing operation
        """
        if session_id in self.sessions:
            self._delete(session_id)
            self._save()

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
        self._save()
        print(f"🗑️  Cleared {count} sessions")
        return {
            'success': True,
            'message': f'Cleared {count} sessions',
            'count': count
        }

    def list_sessions(self):
        """
        List the sessions that have a conversation, most recently used first

        Returns:
            dict: {'sessions': [...], 'total_sessions': n}
        """
        listed = [
            self._summary(session_id, session)
            for session_id, session in self.sessions.items()
            if session['messages']
        ]
        listed.sort(key=lambda s: s['last_activity'], reverse=True)

        return {
            'sessions': listed,
            'total_sessions': len(listed)
        }

    def cleanup_inactive_sessions(self, inactive_hours=24, keep_pinned=True):
        """
        Remove sessions inactive for specified hours

        Args:
            inactive_hours: Hours of inactivity before cleanup
            keep_pinned: Leave pinned sessions alone (default)

        Returns:
            dict: Cleanup results
        """
        cutoff = datetime.now() - timedelta(hours=inactive_hours)

        inactive_sessions = [
            sid for sid, session in self.sessions.items()
            if session['last_activity'] < cutoff and not (keep_pinned and session['pinned'])
        ]

        for session_id in inactive_sessions:
            self._delete(session_id)
        if inactive_sessions:
            self._save()

        print(f"🧹 Cleaned up {len(inactive_sessions)} inactive sessions")

        return {
            'success': True,
            'cleaned': len(inactive_sessions),
            'remaining': len(self.sessions),
            'message': f'Cleaned up {len(inactive_sessions)} inactive sessions'
        }

    # ---------- helpers ----------

    def _retriever(self, session, k):
        search_filter = {'review_id': session['review_id']} if session['review_id'] else None
        return self.vector_store.get_retriever(k=k, filter=search_filter)

    def _chain(self, session):
        """The session's chain, built from its saved messages the first time it is needed"""
        if session['chain'] is None:
            mode = 'code_review' if session['mode'] == 'code_review' else 'docs'
            chain, memory = self.llm_manager.create_conversational_chain(self._retriever(session, session['k']), mode=mode)
            for msg in session['messages']:
                if msg['role'] == 'user':
                    memory.chat_memory.add_user_message(msg['content'])
                else:
                    memory.chat_memory.add_ai_message(msg['content'])
            session['chain'] = chain
            session['memory'] = memory
        return session['chain']

    def _delete(self, session_id):
        session = self.sessions.pop(session_id)
        if session.get('review_id'):
            self.vector_store.delete_by({'review_id': session['review_id']})

    @staticmethod
    def _question_count(session):
        return sum(1 for m in session['messages'] if m['role'] == 'user')

    def _summary(self, session_id, session):
        """What History shows for one session"""
        messages = session['messages']
        is_review = session['mode'] == 'code_review'
        questions = [m['content'] for m in messages if m['role'] == 'user']

        if is_review and messages:
            # The opening summary's first paragraph: "5 issues found: 2 high, 3 low. Most serious: …"
            preview = ' '.join(messages[0]['content'].split('\n\n')[0].split())
        else:
            preview = questions[-1] if questions else ''

        cited = {
            (s.get('metadata') or {}).get('source')
            for m in messages for s in m.get('sources', [])
        }
        cited.discard(None)

        return {
            'session_id': session_id,
            'type': session['mode'],
            'title': session['title'] or ('Code review' if is_review else 'New chat'),
            'preview': _shorten(preview, 200),
            'pinned': session['pinned'],
            'created_at': session['created_at'].isoformat(),
            'last_activity': session['last_activity'].isoformat(),
            'message_count': len(questions),
            'messages': len(messages),
            'source_count': 0 if is_review else len(cited),
            'review_id': session['review_id'],
            'filename': session['filename'],
            'language': session['language'],
        }

    def _save(self):
        if self.store is None:
            return
        records = {
            session_id: self._to_record(session)
            for session_id, session in self.sessions.items()
            if session['messages']  # an empty session has nothing worth keeping
        }
        try:
            self.store.save(records)
        except OSError as e:
            print(f"⚠️  Could not save sessions: {e}")

    @staticmethod
    def _to_record(session):
        return {
            'mode': session['mode'],
            'review_id': session['review_id'],
            'filename': session['filename'],
            'language': session['language'],
            'title': session['title'],
            'pinned': session['pinned'],
            'k': session['k'],
            'created_at': session['created_at'].isoformat(),
            'last_activity': session['last_activity'].isoformat(),
            'messages': [
                {**m, 'timestamp': m['timestamp'].isoformat()} for m in session['messages']
            ],
        }

    @staticmethod
    def _from_record(session_id, record):
        now = datetime.now()
        created = _parse_time(record.get('created_at'), now)
        messages = []
        for m in record.get('messages') or []:
            if m.get('role') not in ('user', 'assistant') or not isinstance(m.get('content'), str):
                continue
            messages.append({**m, 'timestamp': _parse_time(m.get('timestamp'), created)})
        return {
            'chain': None,
            'memory': None,
            'mode': record.get('mode') if record.get('mode') in ('docs', 'code_review') else 'docs',
            'review_id': record.get('review_id'),
            'filename': record.get('filename'),
            'language': record.get('language'),
            'title': record.get('title'),
            'pinned': bool(record.get('pinned')),
            'k': record.get('k') or 5,
            'messages': messages,
            'created_at': created,
            'last_activity': _parse_time(record.get('last_activity'), created),
        }