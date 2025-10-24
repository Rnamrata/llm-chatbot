/**
 * Vector Store and Embedding Module
 * Handles ChromaDB integration and Ollama embeddings
 */

import { Chroma } from 'langchain/vectorstores/chroma';
import { OllamaEmbeddings } from 'langchain/embeddings/ollama';
import { Document } from 'langchain/document';

export class VectorStoreAndEmbedding {
    constructor() {
        // Initialize embeddings
        this.embeddings = new OllamaEmbeddings({
            model: 'nomic-embed-text',
            baseUrl: 'http://localhost:11434',
        });

        // Initialize vector store
        this.vectorstore = null;
        this.initializeVectorStore();
    }

    async initializeVectorStore() {
        try {
            this.vectorstore = await Chroma.fromExistingCollection(this.embeddings, {
                collectionName: 'my_documents',
                url: 'http://localhost:8000',
            });
            console.log('✅ Vector store initialized');
        } catch (error) {
            console.log('Creating new collection...');
            this.vectorstore = await Chroma.fromDocuments([], this.embeddings, {
                collectionName: 'my_documents',
                url: 'http://localhost:8000',
            });
            console.log('✅ New vector store created');
        }
    }

    /**
     * Add chunks to vector store
     * @param {Array} chunks - Array of Document objects or strings
     * @returns {Object} Result with count
     */
    async storeChunks(chunks) {
        if (!chunks || chunks.length === 0) {
            return { error: 'No chunks provided' };
        }

        // Convert strings to Documents if needed
        const documents = [];
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            if (typeof chunk === 'string') {
                // If it's a string, convert to Document
                const doc = new Document({
                    pageContent: chunk,
                    metadata: { chunk_index: i },
                });
                documents.push(doc);
            } else if (chunk instanceof Document || (chunk.pageContent && chunk.metadata)) {
                // Already a Document-like object
                documents.push(chunk);
            } else {
                console.log(`Warning: Skipping invalid chunk type: ${typeof chunk}`);
            }
        }

        try {
            // Ensure vector store is initialized
            if (!this.vectorstore) {
                await this.initializeVectorStore();
            }

            // Add to vector store
            await this.vectorstore.addDocuments(documents);

            console.log(`Stored ${documents.length} chunks in vector database`);
            return {
                data: 'Chunks embedded and stored successfully',
                count: documents.length,
            };
        } catch (error) {
            console.error('Error storing chunks:', error);
            throw error;
        }
    }

    /**
     * Search for similar chunks
     * @param {string} query - Search query
     * @param {number} k - Number of results to return
     * @returns {Array} Array of Document objects
     */
    async search(query, k = 5) {
        try {
            if (!this.vectorstore) {
                await this.initializeVectorStore();
            }

            const results = await this.vectorstore.maxMarginalRelevanceSearch(query, {
                k,
            });
            return results;
        } catch (error) {
            console.error('Error searching:', error);
            return [];
        }
    }

    /**
     * Get count of documents in collection
     * @returns {number} Number of documents
     */
    async getCount() {
        try {
            if (!this.vectorstore) {
                await this.initializeVectorStore();
            }

            // Get collection count
            const collection = await this.vectorstore.collection;
            const count = await collection.count();
            return count;
        } catch (error) {
            console.error('Error getting count:', error);
            return 0;
        }
    }
}

export default VectorStoreAndEmbedding;
