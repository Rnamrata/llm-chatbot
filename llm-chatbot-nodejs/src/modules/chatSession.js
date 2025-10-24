/**
 * Chat Session Module
 * Manages individual chat sessions and conversation history
 */

import { v4 as uuidv4 } from 'uuid';

export class ChatSession {
    /**
     * Initialize Chat Session Manager
     * @param {Object} llmManager - LLMManager instance
     * @param {Object} vectorStore - VectorStoreAndEmbedding instance
     */
    constructor(llmManager, vectorStore) {
        this.llmManager = llmManager;
        this.vectorStore = vectorStore;
        this.sessions = new Map();
    }

    /**
     * Create a new chat session
     * @param {string} sessionId - Optional session ID, generates one if not provided
     * @param {number} k - Number of documents to retrieve
     * @returns {Promise<string>} Session ID
     */
    async createSession(sessionId = null, k = 5) {
        if (!sessionId) {
            sessionId = uuidv4();
        }

        if (this.sessions.has(sessionId)) {
            return sessionId;
        }

        // Create retriever from vector store
        const retriever = this.vectorStore.vectorstore.asRetriever({
            k,
        });

        // Create conversational chain using LLM manager
        const { chain, memory } = await this.llmManager.createConversationalChain(retriever, k);

        // Store session data
        this.sessions.set(sessionId, {
            chain,
            memory,
            created_at: new Date(),
            last_activity: new Date(),
            message_count: 0,
        });

        console.log(`✅ Created new session: ${sessionId}`);
        return sessionId;
    }

    /**
     * Query with conversation history
     * @param {string} question - User's question
     * @param {string} sessionId - Session identifier
     * @param {number} k - Number of documents to retrieve
     * @returns {Promise<Object>} Response with answer, sources, and metadata
     */
    async query(question, sessionId, k = 5) {
        try {
            // Create session if it doesn't exist
            if (!this.sessions.has(sessionId)) {
                await this.createSession(sessionId, k);
            }

            const session = this.sessions.get(sessionId);

            // Run the chain
            const result = await session.chain.call({ question });

            // Update session metadata
            session.last_activity = new Date();
            session.message_count += 1;

            // Format sources using LLM manager
            const sources = this.llmManager.formatSources(
                result.sourceDocuments || []
            );

            return {
                success: true,
                answer: result.answer.trim(),
                sources,
                num_sources: (result.sourceDocuments || []).length,
                session_id: sessionId,
                message_count: session.message_count,
            };
        } catch (error) {
            console.error('Error during query:', error);
            return {
                success: false,
                error: error.message,
                answer: 'An error occurred while processing your query.',
                session_id: sessionId,
            };
        }
    }

    /**
     * Get information about a session
     * @param {string} sessionId - Session identifier
     * @returns {Object} Session information
     */
    getSessionInfo(sessionId) {
        if (!this.sessions.has(sessionId)) {
            return {
                exists: false,
                message: 'Session not found',
            };
        }

        const session = this.sessions.get(sessionId);
        return {
            exists: true,
            session_id: sessionId,
            created_at: session.created_at.toISOString(),
            last_activity: session.last_activity.toISOString(),
            message_count: session.message_count,
        };
    }

    /**
     * Get conversation history for a session
     * @param {string} sessionId - Session identifier
     * @returns {Object} Conversation history
     */
    getHistory(sessionId) {
        if (!this.sessions.has(sessionId)) {
            return {
                history: [],
                length: 0,
                message: 'Session not found',
            };
        }

        const session = this.sessions.get(sessionId);
        const memory = session.memory;
        const messages = memory.chatHistory.messages || [];

        // Format history as Q&A pairs
        const history = [];
        for (let i = 0; i < messages.length; i += 2) {
            if (i + 1 < messages.length) {
                history.push({
                    question: messages[i].content,
                    answer: messages[i + 1].content,
                    timestamp: session.created_at.toISOString(),
                });
            }
        }

        return {
            history,
            length: history.length,
            session_id: sessionId,
            message_count: session.message_count,
        };
    }

    /**
     * Clear conversation history for a session
     * @param {string} sessionId - Session identifier
     * @returns {Object} Result of clearing operation
     */
    clearHistory(sessionId) {
        if (this.sessions.has(sessionId)) {
            this.sessions.delete(sessionId);
            console.log(`🗑️  Cleared session: ${sessionId}`);
            return {
                success: true,
                message: 'Conversation history cleared',
                session_id: sessionId,
            };
        }

        return {
            success: false,
            message: 'Session not found',
            session_id: sessionId,
        };
    }

    /**
     * Clear all sessions
     * @returns {Object} Result of clearing operation
     */
    clearAllSessions() {
        const count = this.sessions.size;
        this.sessions.clear();
        console.log(`🗑️  Cleared ${count} sessions`);
        return {
            success: true,
            message: `Cleared ${count} sessions`,
            count,
        };
    }

    /**
     * List all active sessions
     * @returns {Object} List of session information
     */
    listSessions() {
        const sessionsInfo = [];
        for (const [sessionId, session] of this.sessions.entries()) {
            sessionsInfo.push({
                session_id: sessionId,
                created_at: session.created_at.toISOString(),
                last_activity: session.last_activity.toISOString(),
                message_count: session.message_count,
            });
        }

        return {
            sessions: sessionsInfo,
            total_sessions: sessionsInfo.length,
        };
    }

    /**
     * Remove sessions inactive for specified hours
     * @param {number} inactiveHours - Hours of inactivity before cleanup
     * @returns {Object} Cleanup results
     */
    cleanupInactiveSessions(inactiveHours = 24) {
        const now = new Date();
        const cutoff = new Date(now.getTime() - inactiveHours * 60 * 60 * 1000);

        const inactiveSessions = [];
        for (const [sessionId, session] of this.sessions.entries()) {
            if (session.last_activity < cutoff) {
                inactiveSessions.push(sessionId);
            }
        }

        for (const sessionId of inactiveSessions) {
            this.sessions.delete(sessionId);
        }

        console.log(`🧹 Cleaned up ${inactiveSessions.length} inactive sessions`);

        return {
            success: true,
            cleaned: inactiveSessions.length,
            remaining: this.sessions.size,
            message: `Cleaned up ${inactiveSessions.length} inactive sessions`,
        };
    }
}

export default ChatSession;
