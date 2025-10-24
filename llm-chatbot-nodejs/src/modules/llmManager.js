/**
 * LLM Manager Module
 * Manages LLM initialization and query processing
 */

import { Ollama } from 'langchain/llms/ollama';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { BufferMemory } from 'langchain/memory';
import {
    getReviewPromptTemplate,
    formatCodeContext,
} from './codeReviewPrompts.js';

export class LLMManager {
    /**
     * Initialize LLM Manager
     * @param {string} modelName - Name of the Ollama model to use
     * @param {number} temperature - Temperature for response generation
     */
    constructor(modelName = 'llama3.2', temperature = 0.7) {
        this.modelName = modelName;
        this.temperature = temperature;
        this.llm = this._initializeLLM();
    }

    /**
     * Initialize the Ollama LLM
     * @private
     * @returns {Ollama} Initialized LLM instance
     */
    _initializeLLM() {
        return new Ollama({
            model: this.modelName,
            temperature: this.temperature,
            baseUrl: 'http://localhost:11434',
        });
    }

    /**
     * Create a conversational retrieval chain
     * @param {Object} retriever - Vector store retriever
     * @param {number} k - Number of documents to retrieve
     * @returns {Promise<Object>} Object with chain and memory
     */
    async createConversationalChain(retriever, k = 5) {
        // Create memory for conversation
        const memory = new BufferMemory({
            memoryKey: 'chat_history',
            returnMessages: true,
            outputKey: 'answer',
        });

        // Create the conversational chain
        const chain = ConversationalRetrievalQAChain.fromLLM(
            this.llm,
            retriever,
            {
                memory,
                returnSourceDocuments: true,
                verbose: false,
            }
        );

        return { chain, memory };
    }

    /**
     * Simple query without retrieval (direct LLM call)
     * @param {string} prompt - The prompt to send to LLM
     * @returns {Promise<string>} LLM response
     */
    async simpleQuery(prompt) {
        try {
            const response = await this.llm.call(prompt);
            return response;
        } catch (error) {
            console.error('Error in simple query:', error);
            return `Error: ${error.message}`;
        }
    }

    /**
     * Format source documents for response
     * @param {Array} sourceDocuments - List of retrieved documents
     * @param {number} maxLength - Maximum length for content preview
     * @returns {Array} Formatted source information
     */
    formatSources(sourceDocuments, maxLength = 200) {
        const sources = [];
        for (const doc of sourceDocuments) {
            const content = doc.pageContent;
            const preview = content.length > maxLength
                ? content.substring(0, maxLength) + '...'
                : content;

            const sourceInfo = {
                content: preview,
                metadata: doc.metadata,
                full_length: content.length,
            };
            sources.push(sourceInfo);
        }

        return sources;
    }

    /**
     * Review code directly without RAG retrieval
     * @param {string} code - Code content to review
     * @param {string} language - Programming language
     * @param {string} source - Source filename
     * @param {string} reviewType - Type of review to perform
     * @returns {Promise<string>} Review feedback
     */
    async reviewCodeDirect(code, language, source, reviewType = 'quick') {
        try {
            // Get the appropriate prompt template
            const promptTemplate = getReviewPromptTemplate(reviewType);

            // Format the prompt with code information
            const prompt = await promptTemplate.format({
                code,
                language,
                source,
                context: 'No additional context available (direct review)',
                chat_history: '',
                question: 'Please review this code.',
                start_line: '',
                end_line: '',
            });

            // Get LLM response
            const response = await this.llm.call(prompt);
            return response;
        } catch (error) {
            console.error('Error in reviewCodeDirect:', error);
            return `Error performing code review: ${error.message}`;
        }
    }

    /**
     * Review code with context from related code in the codebase
     * @param {string} code - Code content to review
     * @param {string} language - Programming language
     * @param {string} source - Source filename
     * @param {Array} contextDocuments - Related code documents from vector store
     * @param {string} question - Specific question or review focus
     * @param {string} reviewType - Type of review to perform
     * @returns {Promise<string>} Review feedback
     */
    async reviewCodeWithContext(
        code,
        language,
        source,
        contextDocuments,
        question = 'Please review this code.',
        reviewType = 'comprehensive'
    ) {
        try {
            // Format context from documents
            const context = formatCodeContext(contextDocuments);

            // Get the appropriate prompt template
            const promptTemplate = getReviewPromptTemplate(reviewType);

            // Format the prompt
            const prompt = await promptTemplate.format({
                code,
                language,
                source,
                context,
                chat_history: '',
                question,
                start_line: '',
                end_line: '',
            });

            // Get LLM response
            const response = await this.llm.call(prompt);
            return response;
        } catch (error) {
            console.error('Error in reviewCodeWithContext:', error);
            return `Error performing code review: ${error.message}`;
        }
    }

    /**
     * Explain what code does in clear language
     * @param {string} code - Code content to explain
     * @param {string} language - Programming language
     * @param {string} source - Source filename
     * @param {Array} contextDocuments - Optional related code for context
     * @param {string} question - Specific question about the code
     * @returns {Promise<string>} Code explanation
     */
    async explainCode(
        code,
        language,
        source,
        contextDocuments = null,
        question = 'What does this code do?'
    ) {
        try {
            const context = contextDocuments
                ? formatCodeContext(contextDocuments)
                : 'No additional context';

            const promptTemplate = getReviewPromptTemplate('explanation');

            const prompt = await promptTemplate.format({
                code,
                language,
                source,
                context,
                chat_history: '',
                question,
            });

            const response = await this.llm.call(prompt);
            return response;
        } catch (error) {
            console.error('Error in explainCode:', error);
            return `Error explaining code: ${error.message}`;
        }
    }

    /**
     * Suggest specific improvements for code
     * @param {string} code - Code content to improve
     * @param {string} language - Programming language
     * @param {string} source - Source filename
     * @param {string} goal - Developer's improvement goal
     * @param {Array} contextDocuments - Optional related code for context
     * @returns {Promise<string>} Improvement suggestions
     */
    async suggestImprovements(code, language, source, goal = '', contextDocuments = null) {
        try {
            const context = contextDocuments
                ? formatCodeContext(contextDocuments)
                : 'No additional context';

            const promptTemplate = getReviewPromptTemplate('improvement');

            const question = goal || 'How can I improve this code?';

            const prompt = await promptTemplate.format({
                code,
                language,
                source,
                context,
                chat_history: '',
                question,
            });

            const response = await this.llm.call(prompt);
            return response;
        } catch (error) {
            console.error('Error in suggestImprovements:', error);
            return `Error suggesting improvements: ${error.message}`;
        }
    }

    /**
     * Detect potential bugs in code
     * @param {string} code - Code content to analyze
     * @param {string} language - Programming language
     * @param {string} source - Source filename
     * @param {string} reportedIssue - Reported bug or issue (if any)
     * @param {Array} contextDocuments - Optional related code for context
     * @returns {Promise<string>} Bug analysis
     */
    async detectBugs(
        code,
        language,
        source,
        reportedIssue = '',
        contextDocuments = null
    ) {
        try {
            const context = contextDocuments
                ? formatCodeContext(contextDocuments)
                : 'No additional context';

            const promptTemplate = getReviewPromptTemplate('bug_detection');

            const question = reportedIssue || 'Are there any bugs in this code?';

            const prompt = await promptTemplate.format({
                code,
                language,
                source,
                context,
                chat_history: '',
                question,
            });

            const response = await this.llm.call(prompt);
            return response;
        } catch (error) {
            console.error('Error in detectBugs:', error);
            return `Error detecting bugs: ${error.message}`;
        }
    }
}

export default LLMManager;
