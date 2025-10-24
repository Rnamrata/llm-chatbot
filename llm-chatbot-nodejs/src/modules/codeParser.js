/**
 * Code Parser Module
 * Handles syntax-aware parsing and chunking of source code files.
 * Preserves code structure (functions, classes) while chunking.
 */

import { Document } from 'langchain/document';

export class CodeParser {
    // Supported file extensions and their languages
    static LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.cs': 'csharp',
        '.html': 'html',
        '.css': 'css',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
    };

    /**
     * Initialize the code parser
     * @param {number} maxChunkSize - Maximum size of each code chunk in characters
     */
    constructor(maxChunkSize = 2000) {
        this.maxChunkSize = maxChunkSize;
    }

    /**
     * Detect programming language from file extension
     * @param {string} filename - Name of the file
     * @returns {string} Language name or 'unknown'
     */
    detectLanguage(filename) {
        const extension = filename.includes('.')
            ? '.' + filename.split('.').pop()
            : '';
        return CodeParser.LANGUAGE_EXTENSIONS[extension.toLowerCase()] || 'unknown';
    }

    /**
     * Check if a file is a supported code file
     * @param {string} filename - Name of the file
     * @returns {boolean} True if file is a supported code file
     */
    isCodeFile(filename) {
        return this.detectLanguage(filename) !== 'unknown';
    }

    /**
     * Extract function and class definitions from Python code
     * @param {string} code - Python source code
     * @returns {Array} List of objects with function/class info
     */
    extractPythonFunctions(code) {
        const structures = [];
        const lines = code.split('\n');
        let currentIndent = 0;
        let currentStructure = null;
        let startLine = 0;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // Detect function or class definition
            const match = line.match(/^(\s*)(def|class)\s+(\w+)/);
            if (match) {
                // Save previous structure if exists
                if (currentStructure) {
                    structures.push({
                        type: currentStructure.type,
                        name: currentStructure.name,
                        start_line: startLine,
                        end_line: i - 1,
                        code: lines.slice(startLine, i).join('\n'),
                    });
                }

                // Start new structure
                currentIndent = match[1].length;
                currentStructure = {
                    type: match[2],
                    name: match[3],
                };
                startLine = i;
            } else if (currentStructure && line.trim() &&
                       !line.startsWith(' '.repeat(currentIndent + 1)) &&
                       !line.trim().startsWith('#')) {
                // End of current structure (dedent detected)
                if (line[0] !== ' ' && line[0] !== '\t') {
                    structures.push({
                        type: currentStructure.type,
                        name: currentStructure.name,
                        start_line: startLine,
                        end_line: i - 1,
                        code: lines.slice(startLine, i).join('\n'),
                    });
                    currentStructure = null;
                }
            }
        }

        // Add last structure
        if (currentStructure) {
            structures.push({
                type: currentStructure.type,
                name: currentStructure.name,
                start_line: startLine,
                end_line: lines.length - 1,
                code: lines.slice(startLine).join('\n'),
            });
        }

        return structures;
    }

    /**
     * Extract function and class definitions from JavaScript/TypeScript code
     * @param {string} code - JavaScript/TypeScript source code
     * @returns {Array} List of objects with function/class info
     */
    extractJavaScriptFunctions(code) {
        const structures = [];

        // Match function declarations, arrow functions, and classes
        const patterns = [
            { regex: /function\s+(\w+)\s*\([^)]*\)\s*{/g, type: 'function' },
            { regex: /const\s+(\w+)\s*=\s*\([^)]*\)\s*=>/g, type: 'arrow_function' },
            { regex: /class\s+(\w+)\s*{/g, type: 'class' },
            { regex: /(\w+)\s*:\s*function\s*\([^)]*\)\s*{/g, type: 'method' },
        ];

        for (const { regex, type } of patterns) {
            let match;
            while ((match = regex.exec(code)) !== null) {
                structures.push({
                    type,
                    name: match[1],
                    start: match.index,
                    match: match[0],
                });
            }
        }

        return structures;
    }

    /**
     * Extract import statements from code
     * @param {string} code - Source code
     * @param {string} language - Programming language
     * @returns {Array} List of import statements
     */
    extractImports(code, language) {
        const imports = [];

        if (language === 'python') {
            const matches = code.matchAll(/^(?:from\s+[\w.]+\s+)?import\s+.+$/gm);
            for (const match of matches) {
                imports.push(match[0]);
            }
        } else if (language === 'javascript' || language === 'typescript') {
            const matches = code.matchAll(/^import\s+.+$/gm);
            for (const match of matches) {
                imports.push(match[0]);
            }
        } else if (language === 'java') {
            const matches = code.matchAll(/^import\s+[\w.]+;$/gm);
            for (const match of matches) {
                imports.push(match[0]);
            }
        } else if (language === 'go') {
            // Go import block
            const importBlock = code.match(/import\s+\(([^)]+)\)/);
            if (importBlock) {
                const lines = importBlock[1].split('\n')
                    .map(line => line.trim())
                    .filter(line => line);
                imports.push(...lines);
            } else {
                const matches = code.matchAll(/import\s+"[^"]+"/g);
                for (const match of matches) {
                    imports.push(match[0]);
                }
            }
        }

        return imports;
    }

    /**
     * Chunk code intelligently based on language and structure
     * @param {string} code - Source code content
     * @param {string} filename - Name of the source file
     * @returns {Array} List of Document objects with code chunks
     */
    chunkCode(code, filename) {
        const language = this.detectLanguage(filename);
        const chunks = [];

        // Extract imports
        const imports = this.extractImports(code, language);
        const importsText = imports.join('\n');

        // Language-specific chunking
        if (language === 'python') {
            const structures = this.extractPythonFunctions(code);

            // If we found structures, chunk by function/class
            if (structures.length > 0) {
                for (const struct of structures) {
                    let chunkContent = struct.code;

                    // Include imports if chunk is large enough
                    if (importsText && chunkContent.length + importsText.length < this.maxChunkSize) {
                        chunkContent = importsText + '\n\n' + chunkContent;
                    }

                    chunks.push(new Document({
                        pageContent: chunkContent,
                        metadata: {
                            source: filename,
                            language,
                            content_type: 'code',
                            structure_type: struct.type,
                            structure_name: struct.name,
                            start_line: struct.start_line,
                            end_line: struct.end_line,
                        },
                    }));
                }
            } else {
                // No structures found, chunk by size
                return this._chunkBySize(code, filename, language);
            }
        } else if (language === 'javascript' || language === 'typescript') {
            const structures = this.extractJavaScriptFunctions(code);

            if (structures.length > 0) {
                // For JS/TS, we'll chunk by size but with structure awareness
                return this._chunkBySize(code, filename, language, structures);
            } else {
                return this._chunkBySize(code, filename, language);
            }
        } else {
            // Generic chunking for other languages
            return this._chunkBySize(code, filename, language);
        }

        // If no chunks created, create at least one
        if (chunks.length === 0) {
            chunks.push(new Document({
                pageContent: code,
                metadata: {
                    source: filename,
                    language,
                    content_type: 'code',
                },
            }));
        }

        return chunks;
    }

    /**
     * Chunk code by size with line awareness
     * @param {string} code - Source code
     * @param {string} filename - Filename
     * @param {string} language - Programming language
     * @param {Array} structures - Optional list of code structures
     * @returns {Array} List of Document chunks
     */
    _chunkBySize(code, filename, language, structures = null) {
        const chunks = [];
        const lines = code.split('\n');
        let currentChunk = [];
        let currentSize = 0;
        let startLine = 0;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineSize = line.length + 1; // +1 for newline

            if (currentSize + lineSize > this.maxChunkSize && currentChunk.length > 0) {
                // Save current chunk
                chunks.push(new Document({
                    pageContent: currentChunk.join('\n'),
                    metadata: {
                        source: filename,
                        language,
                        content_type: 'code',
                        start_line: startLine,
                        end_line: i - 1,
                    },
                }));
                currentChunk = [];
                currentSize = 0;
                startLine = i;
            }

            currentChunk.push(line);
            currentSize += lineSize;
        }

        // Add remaining chunk
        if (currentChunk.length > 0) {
            chunks.push(new Document({
                pageContent: currentChunk.join('\n'),
                metadata: {
                    source: filename,
                    language,
                    content_type: 'code',
                    start_line: startLine,
                    end_line: lines.length - 1,
                },
            }));
        }

        return chunks;
    }

    /**
     * Calculate basic code complexity metrics
     * @param {string} code - Source code
     * @param {string} language - Programming language
     * @returns {Object} Dictionary with complexity metrics
     */
    calculateComplexity(code, language) {
        const metrics = {
            lines_of_code: code.split('\n').length,
            num_functions: 0,
            num_classes: 0,
            num_imports: 0,
            cyclomatic_complexity: 0,
        };

        if (language === 'python') {
            metrics.num_functions = (code.match(/^\s*def\s+\w+/gm) || []).length;
            metrics.num_classes = (code.match(/^\s*class\s+\w+/gm) || []).length;
            metrics.num_imports = (code.match(/^(?:from\s+[\w.]+\s+)?import\s+/gm) || []).length;
            // Simple cyclomatic complexity (count decision points)
            metrics.cyclomatic_complexity = (code.match(/\b(if|elif|for|while|except|and|or)\b/g) || []).length;
        } else if (language === 'javascript' || language === 'typescript') {
            metrics.num_functions = (code.match(/function\s+\w+|=>\s*{/g) || []).length;
            metrics.num_classes = (code.match(/class\s+\w+/g) || []).length;
            metrics.num_imports = (code.match(/^import\s+/gm) || []).length;
            metrics.cyclomatic_complexity = (code.match(/\b(if|else if|for|while|case|&&|\|\|)\b/g) || []).length;
        }

        return metrics;
    }
}

export default CodeParser;
