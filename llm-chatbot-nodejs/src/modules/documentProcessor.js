/**
 * Document Processor Module
 * Handles PDF and text file processing, chunking
 */

import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MarkdownTextSplitter } from 'langchain/text_splitter';
import { Document } from 'langchain/document';
import pdfParse from 'pdf-parse';
import fs from 'fs/promises';

export class DocumentProcessor {
    constructor() {
        // Recursive text splitter for general documents
        this.recursiveSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 100,
            separators: ['\n\n\n', '\n\n', '\n', '.', ' ', ''],
        });

        // Markdown splitter for markdown documents
        this.markdownSplitter = new MarkdownTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 100,
        });
    }

    /**
     * Extract text from PDF file
     * @param {string} filePath - Path to PDF file
     * @param {string} fileName - Name of the file
     * @returns {Promise<string>} Extracted text content
     */
    async pdfToText(filePath, fileName) {
        try {
            const dataBuffer = await fs.readFile(filePath);
            const data = await pdfParse(dataBuffer);
            console.log(`PDF: ${fileName} - ${data.numpages} pages`);
            return data.text;
        } catch (error) {
            console.error(`Error reading PDF ${fileName}:`, error);
            return '';
        }
    }

    /**
     * Extract text from text/markdown files
     * @param {string} filePath - Path to text file
     * @param {string} fileName - Name of the file
     * @returns {Promise<string>} File content
     */
    async textFileToText(filePath, fileName) {
        try {
            const content = await fs.readFile(filePath, 'utf-8');
            console.log(`Text file: ${fileName}`);
            return content;
        } catch (error) {
            console.error(`Error reading ${fileName}:`, error);
            return '';
        }
    }

    /**
     * Chunk a single document
     * @param {string} content - Text content to chunk
     * @param {Object} metadata - Metadata for the document
     * @returns {Promise<Array>} Array of Document chunks
     */
    async chunkDocument(content, metadata) {
        if (!content) {
            console.log('No content to chunk');
            return [];
        }

        try {
            // Create Document object
            const doc = new Document({
                pageContent: content,
                metadata,
            });

            // Apply markdown splitting first
            const mdSplits = await this.markdownSplitter.splitDocuments([doc]);

            // Apply recursive splitting
            const finalChunks = await this.recursiveSplitter.splitDocuments(mdSplits);

            console.log(`Created ${finalChunks.length} chunks`);
            return finalChunks;
        } catch (error) {
            console.error('Error chunking document:', error);
            return [];
        }
    }
}

export default DocumentProcessor;
