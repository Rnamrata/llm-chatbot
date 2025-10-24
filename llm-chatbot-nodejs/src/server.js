/**
 * LLM Chatbot Server - Node.js/Express Version
 * RAG-based chatbot with code review capabilities
 */

import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import dotenv from 'dotenv';

// Import modules
import { DocumentProcessor } from './modules/documentProcessor.js';
import { VectorStoreAndEmbedding } from './modules/vectorStoreAndEmbedding.js';
import { FileManager } from './modules/fileManager.js';
import { LLMManager } from './modules/llmManager.js';
import { ChatSession } from './modules/chatSession.js';
import { CodeParser } from './modules/codeParser.js';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure multer for file uploads
const upload = multer({ storage: multer.memoryStorage() });

// Initialize modules
const documentProcessor = new DocumentProcessor();
const vectorStore = new VectorStoreAndEmbedding();
const fileManager = new FileManager(documentProcessor, vectorStore);
const llmManager = new LLMManager('llama3.2', 0.7);
const chatManager = new ChatSession(llmManager, vectorStore);

// ==================== UPLOAD ENDPOINTS ====================

/**
 * Upload a file from device
 * POST /upload/file
 */
app.post('/upload/file', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file provided' });
        }

        const result = await fileManager.uploadFile(req.file);
        const statusCode = result.success ? 200 : 400;
        res.status(statusCode).json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Upload YouTube video
 * POST /upload/youtube
 */
app.post('/upload/youtube', async (req, res) => {
    try {
        const url = req.body.url || req.query.url;
        const result = await fileManager.uploadMediaFile(url);
        const statusCode = result.success ? 200 : 400;
        res.status(statusCode).json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Upload web page content
 * POST /upload/web
 */
app.post('/upload/web', async (req, res) => {
    try {
        const url = req.body.url || req.query.url;
        const result = await fileManager.webFileUpload(url);
        const statusCode = result.success ? 200 : 400;
        res.status(statusCode).json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Upload code file for review
 * POST /upload/code
 */
app.post('/upload/code', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file provided' });
        }

        const result = await fileManager.uploadCodeForReview(req.file);
        const statusCode = result.success ? 200 : 400;
        res.status(statusCode).json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// ==================== CODE REVIEW ENDPOINTS ====================

/**
 * Quick code review focusing on critical issues
 * POST /review/quick
 */
app.post('/review/quick', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file provided' });
        }

        const filename = req.file.originalname;
        const codeContent = req.file.buffer.toString('utf-8');

        // Detect language
        const parser = new CodeParser();
        const language = parser.detectLanguage(filename);

        if (language === 'unknown') {
            return res.status(400).json({ error: 'Unsupported file type' });
        }

        // Perform quick review
        const review = await llmManager.reviewCodeDirect(
            codeContent,
            language,
            filename,
            'quick'
        );

        res.json({
            success: true,
            filename,
            language,
            review_type: 'quick',
            review,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Comprehensive code review with detailed analysis
 * POST /review/comprehensive
 */
app.post('/review/comprehensive', upload.single('file'), async (req, res) => {
    try {
        let codeContent, filename, question;

        // Handle file upload
        if (req.file) {
            filename = req.file.originalname;
            codeContent = req.file.buffer.toString('utf-8');
            question = req.body.question || 'Please provide a comprehensive review.';
        }
        // Handle JSON request
        else if (req.body.code) {
            codeContent = req.body.code;
            filename = req.body.filename || 'code_snippet';
            question = req.body.question || 'Please provide a comprehensive review.';
        } else {
            return res.status(400).json({ error: 'No code provided' });
        }

        // Detect language
        const parser = new CodeParser();
        const language = parser.detectLanguage(filename);

        // Get related code from vector store for context
        const contextDocs = await vectorStore.search(codeContent.substring(0, 500), 3);

        // Perform comprehensive review
        const review = await llmManager.reviewCodeWithContext(
            codeContent,
            language,
            filename,
            contextDocs,
            question,
            'comprehensive'
        );

        res.json({
            success: true,
            filename,
            language,
            review_type: 'comprehensive',
            review,
            context_used: contextDocs.length,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Security-focused code review
 * POST /review/security
 */
app.post('/review/security', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file provided' });
        }

        const filename = req.file.originalname;
        const codeContent = req.file.buffer.toString('utf-8');

        const parser = new CodeParser();
        const language = parser.detectLanguage(filename);

        // Get context
        const contextDocs = await vectorStore.search(codeContent.substring(0, 500), 3);

        // Perform security review
        const review = await llmManager.reviewCodeWithContext(
            codeContent,
            language,
            filename,
            contextDocs,
            'Analyze for security vulnerabilities.',
            'security'
        );

        res.json({
            success: true,
            filename,
            language,
            review_type: 'security',
            review,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Performance-focused code review
 * POST /review/performance
 */
app.post('/review/performance', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file provided' });
        }

        const filename = req.file.originalname;
        const codeContent = req.file.buffer.toString('utf-8');

        const parser = new CodeParser();
        const language = parser.detectLanguage(filename);

        // Get context
        const contextDocs = await vectorStore.search(codeContent.substring(0, 500), 3);

        // Perform performance review
        const review = await llmManager.reviewCodeWithContext(
            codeContent,
            language,
            filename,
            contextDocs,
            'Analyze for performance issues and optimization opportunities.',
            'performance'
        );

        res.json({
            success: true,
            filename,
            language,
            review_type: 'performance',
            review,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Explain what code does
 * POST /review/explain
 */
app.post('/review/explain', upload.single('file'), async (req, res) => {
    try {
        let codeContent, filename, question;

        // Handle file upload
        if (req.file) {
            filename = req.file.originalname;
            codeContent = req.file.buffer.toString('utf-8');
            question = req.body.question || 'What does this code do?';
        }
        // Handle JSON request
        else if (req.body.code) {
            codeContent = req.body.code;
            filename = req.body.filename || 'code_snippet';
            question = req.body.question || 'What does this code do?';
        } else {
            return res.status(400).json({ error: 'No code provided' });
        }

        const parser = new CodeParser();
        const language = parser.detectLanguage(filename);

        // Get context
        const contextDocs = await vectorStore.search(codeContent.substring(0, 500), 3);

        // Explain code
        const explanation = await llmManager.explainCode(
            codeContent,
            language,
            filename,
            contextDocs,
            question
        );

        res.json({
            success: true,
            filename,
            language,
            explanation,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Detect potential bugs in code
 * POST /review/bugs
 */
app.post('/review/bugs', upload.single('file'), async (req, res) => {
    try {
        let codeContent, filename, issue;

        // Handle file upload
        if (req.file) {
            filename = req.file.originalname;
            codeContent = req.file.buffer.toString('utf-8');
            issue = req.body.issue || '';
        }
        // Handle JSON request
        else if (req.body.code) {
            codeContent = req.body.code;
            filename = req.body.filename || 'code_snippet';
            issue = req.body.issue || '';
        } else {
            return res.status(400).json({ error: 'No code provided' });
        }

        const parser = new CodeParser();
        const language = parser.detectLanguage(filename);

        // Get context
        const contextDocs = await vectorStore.search(codeContent.substring(0, 500), 3);

        // Detect bugs
        const analysis = await llmManager.detectBugs(
            codeContent,
            language,
            filename,
            issue,
            contextDocs
        );

        res.json({
            success: true,
            filename,
            language,
            bug_analysis: analysis,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Suggest code improvements
 * POST /review/improve
 */
app.post('/review/improve', upload.single('file'), async (req, res) => {
    try {
        let codeContent, filename, goal;

        // Handle file upload
        if (req.file) {
            filename = req.file.originalname;
            codeContent = req.file.buffer.toString('utf-8');
            goal = req.body.goal || '';
        }
        // Handle JSON request
        else if (req.body.code) {
            codeContent = req.body.code;
            filename = req.body.filename || 'code_snippet';
            goal = req.body.goal || '';
        } else {
            return res.status(400).json({ error: 'No code provided' });
        }

        const parser = new CodeParser();
        const language = parser.detectLanguage(filename);

        // Get context
        const contextDocs = await vectorStore.search(codeContent.substring(0, 500), 3);

        // Suggest improvements
        const suggestions = await llmManager.suggestImprovements(
            codeContent,
            language,
            filename,
            goal,
            contextDocs
        );

        res.json({
            success: true,
            filename,
            language,
            suggestions,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// ==================== CHAT ENDPOINTS ====================

/**
 * Chat with your documents
 * POST /chat
 */
app.post('/chat', async (req, res) => {
    try {
        if (!req.body.question) {
            return res.status(400).json({ error: 'No question provided' });
        }

        const question = req.body.question;
        const sessionId = req.body.session_id || uuidv4();
        const k = req.body.k || 5;

        const result = await chatManager.query(question, sessionId, k);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Start a new chat session
 * POST /chat/new
 */
app.post('/chat/new', async (req, res) => {
    try {
        const sessionId = uuidv4();
        await chatManager.createSession(sessionId);
        res.json({
            session_id: sessionId,
            message: 'New chat session created',
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Get information about a specific session
 * GET /chat/session/:sessionId
 */
app.get('/chat/session/:sessionId', (req, res) => {
    try {
        const info = chatManager.getSessionInfo(req.params.sessionId);
        res.json(info);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Get chat history for a specific session
 * GET /chat/history/:sessionId
 */
app.get('/chat/history/:sessionId', (req, res) => {
    try {
        const history = chatManager.getHistory(req.params.sessionId);
        res.json(history);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Clear chat history for a specific session
 * DELETE /chat/clear/:sessionId
 */
app.delete('/chat/clear/:sessionId', (req, res) => {
    try {
        const result = chatManager.clearHistory(req.params.sessionId);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * List all active chat sessions
 * GET /chat/sessions
 */
app.get('/chat/sessions', (req, res) => {
    try {
        const result = chatManager.listSessions();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Clean up inactive sessions
 * POST /chat/cleanup
 */
app.post('/chat/cleanup', (req, res) => {
    try {
        const inactiveHours = req.body.inactive_hours || 24;
        const result = chatManager.cleanupInactiveSessions(inactiveHours);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// ==================== UTILITY ENDPOINTS ====================

/**
 * Get statistics about the vector database
 * GET /stats
 */
app.get('/stats', async (req, res) => {
    try {
        const count = await vectorStore.getCount();
        const sessions = chatManager.listSessions();

        res.json({
            total_chunks: count,
            total_sessions: sessions.total_sessions,
            status: count > 0 ? 'ready' : 'empty',
            message: `Vector database contains ${count} chunks`,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Health check endpoint
 * GET /health
 */
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        service: 'RAG System',
        version: '1.0',
        llm_model: llmManager.modelName,
    });
});

// ==================== ERROR HANDLERS ====================

app.use((req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

app.use((error, req, res, next) => {
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// ==================== START SERVER ====================

app.listen(PORT, () => {
    console.log(`🌐 Server starting on http://0.0.0.0:${PORT}`);
    console.log(`📚 RAG Chatbot System Ready`);
    console.log(`🤖 LLM Model: ${llmManager.modelName}`);
});

export default app;
