/**
 * File Manager Module
 * Handles file uploads, YouTube videos, web scraping, and code review uploads
 */

import fs from 'fs/promises';
import path from 'path';
import ytdl from 'ytdl-core';
import { pipeline } from '@xenova/transformers';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { Document } from 'langchain/document';
import { CodeParser } from './codeParser.js';

export class FileManager {
    /**
     * Initialize FileManager with dependencies
     * @param {Object} documentProcessor - DocumentProcessor instance
     * @param {Object} vectorStore - VectorStoreAndEmbedding instance
     * @param {Object} codeParser - CodeParser instance (optional)
     */
    constructor(documentProcessor, vectorStore, codeParser = null) {
        this.documentProcessor = documentProcessor;
        this.vectorStore = vectorStore;
        this.codeParser = codeParser || new CodeParser();
    }

    /**
     * Upload file from device (documents or code)
     * @param {Object} file - Uploaded file object
     * @returns {Promise<Object>} Result object
     */
    async uploadFile(file) {
        try {
            const filename = file.originalname;
            const destination = path.join('uploads', filename);

            // Save file
            await fs.writeFile(destination, file.buffer);

            // Check if it's a code file
            if (this.codeParser.isCodeFile(filename)) {
                // Handle as code file
                const codeContent = file.buffer.toString('utf-8');

                // Use code-specific chunking
                const chunks = this.codeParser.chunkCode(codeContent, filename);

                if (!chunks || chunks.length === 0) {
                    return { error: 'Failed to parse code file', success: false };
                }

                // Store in vector database
                const storeResult = await this.vectorStore.storeChunks(chunks);

                const language = this.codeParser.detectLanguage(filename);

                return {
                    success: true,
                    message: 'Code file uploaded and processed successfully',
                    filename,
                    file_type: 'code',
                    language,
                    chunks_created: storeResult.count,
                };
            }

            // Handle as document file
            let content;
            if (filename.endsWith('.pdf')) {
                content = await this.documentProcessor.pdfToText(destination, filename);
            } else if (filename.endsWith('.txt') || filename.endsWith('.md')) {
                content = await this.documentProcessor.textFileToText(destination, filename);
            } else {
                return { error: 'Unsupported file type', success: false };
            }

            if (!content) {
                return { error: 'No content extracted from file', success: false };
            }

            // Chunk the content
            const chunks = await this.documentProcessor.chunkDocument(content, {
                source: filename,
                type: 'file_upload',
                content_type: 'document',
            });

            if (!chunks || chunks.length === 0) {
                return { error: 'Failed to create chunks', success: false };
            }

            // Store in vector database
            const storeResult = await this.vectorStore.storeChunks(chunks);

            return {
                success: true,
                message: 'File uploaded and processed successfully',
                filename,
                file_type: 'document',
                chunks_created: storeResult.count,
            };
        } catch (error) {
            console.error('Error in uploadFile:', error);
            return { error: error.message, success: false };
        }
    }

    /**
     * Upload code file specifically for code review
     * @param {Object} file - Uploaded file object
     * @returns {Promise<Object>} Result object
     */
    async uploadCodeForReview(file) {
        try {
            const filename = file.originalname;
            const destination = path.join('uploads', filename);

            // Save file
            await fs.writeFile(destination, file.buffer);

            // Verify it's a code file
            if (!this.codeParser.isCodeFile(filename)) {
                return { error: 'File is not a supported code file', success: false };
            }

            // Read code content
            const codeContent = file.buffer.toString('utf-8');

            const language = this.codeParser.detectLanguage(filename);

            // Calculate complexity metrics
            const complexity = this.codeParser.calculateComplexity(codeContent, language);

            // Chunk the code
            const chunks = this.codeParser.chunkCode(codeContent, filename);

            if (!chunks || chunks.length === 0) {
                return { error: 'Failed to parse code file', success: false };
            }

            // Store in vector database
            const storeResult = await this.vectorStore.storeChunks(chunks);

            return {
                success: true,
                message: 'Code file uploaded for review successfully',
                filename,
                language,
                chunks_created: storeResult.count,
                complexity,
            };
        } catch (error) {
            console.error('Error in uploadCodeForReview:', error);
            return { error: error.message, success: false };
        }
    }

    /**
     * Download YouTube audio file
     * @param {string} saveDir - Directory to save the file
     * @param {string} url - YouTube URL
     * @returns {Promise<string|null>} Path to downloaded audio file
     */
    async downloadYouTubeFile(saveDir, url) {
        try {
            // Create save directory if it doesn't exist
            await fs.mkdir(saveDir, { recursive: true });

            // Get video info
            const info = await ytdl.getInfo(url);
            const title = info.videoDetails.title.replace(/[^\w\s]/gi, '');
            const audioPath = path.join(saveDir, `${title}.mp3`);

            // Download audio
            const audioStream = ytdl(url, {
                quality: 'highestaudio',
                filter: 'audioonly',
            });

            const writeStream = (await import('fs')).createWriteStream(audioPath);

            audioStream.pipe(writeStream);

            return new Promise((resolve, reject) => {
                writeStream.on('finish', () => resolve(audioPath));
                writeStream.on('error', reject);
            });
        } catch (error) {
            console.error('Error downloading YouTube file:', error);
            return null;
        }
    }

    /**
     * Transcribe audio file using Whisper
     * @param {string} audioFilePath - Path to audio file
     * @param {string} url - Original URL
     * @returns {Promise<Document>} Transcribed document
     */
    async transcribeAudioFile(audioFilePath, url) {
        try {
            // Initialize Whisper pipeline
            const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base');

            console.log(`Transcribing ${audioFilePath}...`);

            // Read audio file
            const audioBuffer = await fs.readFile(audioFilePath);

            // Transcribe
            const result = await transcriber(audioBuffer);

            const doc = new Document({
                pageContent: result.text,
                metadata: {
                    source: url,
                    file: path.basename(audioFilePath),
                },
            });

            return doc;
        } catch (error) {
            console.error('Error transcribing audio:', error);
            throw error;
        }
    }

    /**
     * Upload YouTube video
     * @param {string} url - YouTube URL
     * @returns {Promise<Object>} Result object
     */
    async uploadMediaFile(url) {
        try {
            if (!url) {
                return { error: 'No URL provided', success: false };
            }

            const saveDir = 'uploads/media/';
            await fs.mkdir(saveDir, { recursive: true });

            // Download audio
            const audioFilePath = await this.downloadYouTubeFile(saveDir, url);

            if (!audioFilePath) {
                return { error: 'Failed to download audio from YouTube', success: false };
            }

            // Transcribe audio
            const doc = await this.transcribeAudioFile(audioFilePath, url);

            // Save transcription to text file
            const baseFilename = path.basename(audioFilePath, path.extname(audioFilePath));
            const transcriptionFilename = `${baseFilename}_transcription.txt`;
            const transcriptionPath = path.join('uploads', transcriptionFilename);

            await fs.writeFile(transcriptionPath, doc.pageContent, 'utf-8');

            // Chunk the transcription
            const chunks = await this.documentProcessor.chunkDocument(doc.pageContent, {
                source: url,
                type: 'youtube',
                filename: transcriptionFilename,
            });

            if (!chunks || chunks.length === 0) {
                return { error: 'Failed to create chunks', success: false };
            }

            // Store in vector database
            const storeResult = await this.vectorStore.storeChunks(chunks);

            return {
                success: true,
                message: 'YouTube video processed and stored successfully',
                url,
                transcription_file: transcriptionFilename,
                chunks_created: storeResult.count,
            };
        } catch (error) {
            console.error('Error in uploadMediaFile:', error);
            return { error: error.message, success: false };
        }
    }

    /**
     * Upload web page content
     * @param {string} url - Web page URL
     * @returns {Promise<Object>} Result object
     */
    async webFileUpload(url) {
        try {
            if (!url) {
                return { error: 'No URL provided', success: false };
            }

            // Load web content
            const loader = new CheerioWebBaseLoader(url);
            const data = await loader.load();

            if (!data || data.length === 0) {
                return { error: 'Failed to load data from URL', success: false };
            }

            const doc = data[0];

            // Create a safe filename from the title
            const title = doc.metadata.title || 'web_content';
            let safeFilename = title.replace(/\s+/g, '_').replace(/[/\\]/g, '_');
            safeFilename = safeFilename.replace(/[^a-zA-Z0-9_.-]/g, '');

            // Save the page content to a text file
            const destination = path.join('uploads', `${safeFilename}.txt`);

            const fileContent = `Source: ${doc.metadata.source || 'N/A'}\n` +
                `Title: ${doc.metadata.title || 'N/A'}\n` +
                `Description: ${doc.metadata.description || 'N/A'}\n` +
                `Language: ${doc.metadata.language || 'N/A'}\n\n` +
                '='.repeat(80) + '\n\n' +
                doc.pageContent;

            await fs.writeFile(destination, fileContent, 'utf-8');

            console.log(`Web content saved to: ${destination}`);

            // Chunk the content
            const chunks = await this.documentProcessor.chunkDocument(doc.pageContent, {
                source: url,
                type: 'web',
                title,
                filename: safeFilename,
            });

            if (!chunks || chunks.length === 0) {
                return { error: 'Failed to create chunks', success: false };
            }

            // Store in vector database
            const storeResult = await this.vectorStore.storeChunks(chunks);

            return {
                success: true,
                message: 'Web page processed and stored successfully',
                url,
                filename: `${safeFilename}.txt`,
                chunks_created: storeResult.count,
            };
        } catch (error) {
            console.error('Error in webFileUpload:', error);
            return { error: error.message, success: false };
        }
    }
}

export default FileManager;
