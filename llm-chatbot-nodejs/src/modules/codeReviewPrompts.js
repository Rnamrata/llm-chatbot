/**
 * ============================================================================
 * CODE REVIEW PROMPTS MODULE
 * ============================================================================
 *
 * PURPOSE:
 * This module contains all the specialized prompt templates used for different
 * types of code reviews. Prompts are carefully crafted instructions sent to the
 * LLM (Large Language Model) to guide its analysis and response generation.
 *
 * HOW IT WORKS:
 * 1. Each prompt template is a string with placeholders (e.g., {code}, {language})
 * 2. LangChain's PromptTemplate class converts these strings into reusable templates
 * 3. When needed, placeholders are replaced with actual values (code, filename, etc.)
 * 4. The completed prompt is sent to the LLM for processing
 * 5. The LLM analyzes the code according to the prompt instructions
 *
 * PROMPT TYPES:
 * - Comprehensive: Full detailed analysis covering all aspects
 * - Quick: Fast review focusing only on critical issues
 * - Security: OWASP and security-focused analysis
 * - Performance: Algorithm complexity and optimization
 * - Best Practices: Design patterns and code quality
 * - Conversational: Interactive Q&A about code
 * - Explanation: Educational code walkthrough
 * - Improvement: Refactoring suggestions
 * - Bug Detection: Finding potential errors
 * - Comparison: Analyzing code changes
 *
 * @module codeReviewPrompts
 */

// Import LangChain's PromptTemplate class for creating reusable prompt templates
import { PromptTemplate } from 'langchain/prompts';

/**
 * COMPREHENSIVE CODE REVIEW PROMPT
 *
 * This is the most detailed review type, covering all aspects of code quality.
 * It instructs the LLM to act as an expert code reviewer and analyze:
 * - Code quality and readability
 * - Best practices and design patterns
 * - Potential bugs and edge cases
 * - Security vulnerabilities
 * - Performance concerns
 * - Testing and maintainability
 *
 * Template Variables:
 * {code} - The actual source code to review
 * {language} - Programming language (e.g., "javascript", "python")
 * {source} - Filename or source identifier
 * {start_line} - Starting line number of code snippet
 * {end_line} - Ending line number of code snippet
 * {context} - Related code from the codebase for reference
 * {chat_history} - Previous conversation for continuity
 * {question} - Specific question or focus area from user
 */
const COMPREHENSIVE_CODE_REVIEW_PROMPT = `You are an expert code reviewer with years of experience in software engineering best practices, security, and performance optimization.

Review the following code and provide detailed, actionable feedback:

**Code to Review:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}
**Lines:** {start_line}-{end_line}

**Context from codebase:**
{context}

**Previous conversation:**
{chat_history}

**Specific Question (if any):**
{question}

Please analyze the code for:

1. **Code Quality & Readability**
   - Variable and function naming
   - Code organization and structure
   - Comments and documentation
   - Code duplication

2. **Best Practices & Design Patterns**
   - Language-specific idioms
   - Design patterns usage
   - SOLID principles
   - DRY, KISS, YAGNI principles

3. **Potential Bugs**
   - Logic errors
   - Edge cases handling
   - Null/undefined checks
   - Error handling

4. **Security Issues**
   - Input validation
   - SQL injection risks
   - XSS vulnerabilities
   - Authentication/authorization issues
   - Sensitive data exposure

5. **Performance Concerns**
   - Algorithm efficiency
   - Memory usage
   - Database queries optimization
   - Resource leaks

6. **Testing & Maintainability**
   - Testability of code
   - Test coverage needs
   - Ease of modification
   - Technical debt

Provide specific, actionable feedback with:
- Line references where applicable
- Severity level (Critical/High/Medium/Low)
- Concrete suggestions for improvement
- Code examples for fixes when helpful

Format your response in a clear, structured manner.
`;

// Quick code review prompt (faster, less detailed)
const QUICK_CODE_REVIEW_PROMPT = `You are a code reviewer. Quickly analyze this code for critical issues:

**Code:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}

Focus on:
1. Critical bugs or security issues
2. Major performance problems
3. Obvious best practice violations

Provide concise, actionable feedback with line references.

Answer:`;

// Security review prompt
const SECURITY_REVIEW_PROMPT = `You are a security expert. Review this code specifically for security vulnerabilities:

**Code:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}

Check for:
- Input validation issues
- SQL injection vulnerabilities
- XSS vulnerabilities
- Authentication/authorization flaws
- Sensitive data exposure
- Insecure dependencies
- Cryptography misuse
- OWASP Top 10 issues

Provide detailed security findings with severity ratings.

Answer:`;

// Performance review prompt
const PERFORMANCE_REVIEW_PROMPT = `You are a performance optimization expert. Review this code for performance issues:

**Code:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}

Analyze:
- Algorithm complexity (Big O)
- Memory usage and leaks
- Database query optimization
- I/O operations efficiency
- Caching opportunities
- Concurrency issues
- Resource management

Provide specific optimization suggestions.

Answer:`;

// Best practices prompt
const BEST_PRACTICES_PROMPT = `You are a software architecture expert. Review this code for best practices and design patterns:

**Code:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}
**Language:** {language}

Evaluate:
- Design patterns usage
- SOLID principles adherence
- Language-specific idioms
- Code organization
- Separation of concerns
- Dependency management

Provide recommendations for improvement.

Answer:`;

// Conversational code review with context
const CONVERSATIONAL_CODE_REVIEW_PROMPT = `You are an experienced code reviewer having a conversation with a developer about their code.

**Code being discussed:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}

**Related code from codebase (for context):**
{context}

**Previous conversation:**
{chat_history}

**Developer's question:**
{question}

Provide a helpful, conversational response that:
- Directly answers their question
- References specific lines of code
- Explains the reasoning behind your suggestions
- Offers alternative approaches when applicable
- Maintains context from previous messages

Keep your tone professional but friendly, like a senior developer helping a colleague.

Answer:`;

// Code explanation prompt
const CODE_EXPLANATION_PROMPT = `You are a senior developer explaining code to help someone understand it better.

**Code:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}
**Context:** {context}

**Question:**
{question}

Explain:
- What the code does
- How it works (step by step if needed)
- Why certain approaches were used
- Potential gotchas or edge cases
- How it fits into the larger system

Use clear, accessible language with examples when helpful.

Answer:`;

// Code improvement suggestions
const CODE_IMPROVEMENT_PROMPT = `You are a code quality expert. Suggest specific improvements for this code:

**Current Code:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}

**Developer's goal:**
{question}

Provide:
1. Specific code improvements
2. Refactoring suggestions
3. Alternative approaches
4. Updated code examples
5. Trade-offs of each approach

Focus on practical, implementable suggestions.

Answer:`;

// Bug detection prompt
const BUG_DETECTION_PROMPT = `You are a debugging expert. Analyze this code for potential bugs:

**Code:**
\`\`\`{language}
{code}
\`\`\`

**Source:** {source}
**Context:** {context}

**Reported issue (if any):**
{question}

Identify:
- Logic errors
- Edge cases not handled
- Race conditions
- Off-by-one errors
- Null/undefined issues
- Type mismatches
- Error handling gaps

For each bug found:
- Describe the issue
- Explain why it's a problem
- Provide a fix
- Suggest test cases

Answer:`;

// Code comparison prompt
const CODE_COMPARISON_PROMPT = `You are reviewing code changes. Compare the before and after:

**Original Code:**
\`\`\`{language}
{original_code}
\`\`\`

**Modified Code:**
\`\`\`{language}
{modified_code}
\`\`\`

**Source:** {source}

**Change description:**
{question}

Evaluate:
- Whether changes improve code quality
- If any regressions were introduced
- Test coverage impact
- Documentation updates needed
- Potential side effects

Provide feedback on the changes.

Answer:`;

/**
 * GET REVIEW PROMPT TEMPLATE
 *
 * This function selects and returns the appropriate prompt template based on
 * the requested review type. It acts as a factory function that maps review
 * type strings to their corresponding prompt templates.
 *
 * HOW IT WORKS:
 * 1. Receives a reviewType string (e.g., "comprehensive", "security")
 * 2. Looks up the corresponding prompt template from the prompts object
 * 3. If review type not found, defaults to comprehensive review
 * 4. Wraps the prompt string in a LangChain PromptTemplate object
 * 5. Returns the template ready for variable substitution
 *
 * EXAMPLE USAGE:
 * const template = getReviewPromptTemplate('security');
 * const prompt = await template.format({
 *     code: 'function login(user, pass) { ... }',
 *     language: 'javascript',
 *     source: 'auth.js',
 *     context: 'Related auth code...',
 *     chat_history: '',
 *     question: 'Is this secure?'
 * });
 * // Now 'prompt' contains the full prompt ready to send to the LLM
 *
 * @param {string} reviewType - Type of review (comprehensive, quick, security, etc.)
 * @returns {PromptTemplate} LangChain PromptTemplate object configured for the review type
 */
export function getReviewPromptTemplate(reviewType = 'comprehensive') {
    // Map of review type names to their corresponding prompt templates
    // This allows easy lookup and addition of new review types
    const prompts = {
        comprehensive: COMPREHENSIVE_CODE_REVIEW_PROMPT,      // Full detailed analysis
        quick: QUICK_CODE_REVIEW_PROMPT,                      // Fast critical issues only
        security: SECURITY_REVIEW_PROMPT,                     // Security vulnerabilities
        performance: PERFORMANCE_REVIEW_PROMPT,               // Performance optimization
        best_practices: BEST_PRACTICES_PROMPT,                // Design patterns & quality
        conversational: CONVERSATIONAL_CODE_REVIEW_PROMPT,    // Interactive Q&A
        explanation: CODE_EXPLANATION_PROMPT,                 // Educational walkthrough
        improvement: CODE_IMPROVEMENT_PROMPT,                 // Refactoring suggestions
        bug_detection: BUG_DETECTION_PROMPT,                  // Bug finding
        comparison: CODE_COMPARISON_PROMPT,                   // Code diff analysis
    };

    // Select the template based on reviewType, or use comprehensive as default
    // The || operator ensures we always have a valid template
    const template = prompts[reviewType] || COMPREHENSIVE_CODE_REVIEW_PROMPT;

    // Create and return a LangChain PromptTemplate object
    // inputVariables defines which placeholders the template expects
    // These MUST match the {variable} names in the template strings
    return new PromptTemplate({
        template,                                              // The prompt string
        inputVariables: ['code', 'language', 'source', 'context', 'chat_history', 'question'],
    });
}

/**
 * FORMAT CODE CONTEXT
 *
 * This function takes documents retrieved from the vector database and formats
 * them into a readable context string to send along with the code being reviewed.
 * This provides the LLM with additional related code for better analysis.
 *
 * HOW IT WORKS:
 * 1. Receives an array of Document objects from vector similarity search
 * 2. Extracts metadata (source file, language, line numbers) from each document
 * 3. Formats each document as a markdown code block with source info
 * 4. Concatenates documents until maxContextLength is reached
 * 5. Returns formatted string ready to insert into prompt
 *
 * WHY THIS IS NEEDED:
 * When reviewing code, having context from related code helps the LLM:
 * - Understand how the code fits into the larger system
 * - Identify inconsistencies with coding patterns used elsewhere
 * - Suggest improvements that align with existing code style
 * - Detect potential integration issues
 *
 * EXAMPLE OUTPUT:
 * "
 * **From utils.js (lines 45-60):**
 * ```javascript
 * function validateInput(data) { ... }
 * ```
 *
 * **From auth.js (lines 12-25):**
 * ```javascript
 * function checkPermissions(user) { ... }
 * ```
 * "
 *
 * @param {Array} documents - Array of Document objects from vector store search
 * @param {number} maxContextLength - Maximum total characters for all context (default: 1500)
 * @returns {string} Formatted markdown string with code context
 */
export function formatCodeContext(documents, maxContextLength = 1500) {
    // Handle edge case: no documents provided
    if (!documents || documents.length === 0) {
        return 'No additional context available.';
    }

    const contextParts = [];      // Array to store formatted document strings
    let currentLength = 0;         // Track total length to respect maxContextLength

    // Iterate through each document and format it
    for (const doc of documents) {
        // Extract metadata with safe fallback values using optional chaining (?.)
        // If metadata fields don't exist, use default values
        const source = doc.metadata?.source || 'unknown';
        const language = doc.metadata?.language || 'unknown';
        const startLine = doc.metadata?.start_line || '';
        const endLine = doc.metadata?.end_line || '';

        // Format line number information (only if we have line numbers)
        const lineInfo = startLine ? `(lines ${startLine}-${endLine})` : '';

        // Truncate document content to 500 chars to keep context focused
        const content = doc.pageContent.substring(0, 500);

        // Format as markdown code block with source information
        // Template: **From <file> <lines>:**\n```<language>\n<code>\n```
        const part = `\n**From ${source} ${lineInfo}:**\n\`\`\`${language}\n${content}\n\`\`\`\n`;

        // Check if adding this document would exceed max length
        if (currentLength + part.length > maxContextLength) {
            break;  // Stop adding more documents to respect size limit
        }

        // Add this formatted document to our collection
        contextParts.push(part);
        currentLength += part.length;
    }

    // If we couldn't include any documents (all too large), return no context message
    if (contextParts.length === 0) {
        return 'No additional context available.';
    }

    // Join all formatted documents with newlines and return
    return contextParts.join('\n');
}

// Export all prompts for testing or direct use
export {
    COMPREHENSIVE_CODE_REVIEW_PROMPT,
    QUICK_CODE_REVIEW_PROMPT,
    SECURITY_REVIEW_PROMPT,
    PERFORMANCE_REVIEW_PROMPT,
    BEST_PRACTICES_PROMPT,
    CONVERSATIONAL_CODE_REVIEW_PROMPT,
    CODE_EXPLANATION_PROMPT,
    CODE_IMPROVEMENT_PROMPT,
    BUG_DETECTION_PROMPT,
    CODE_COMPARISON_PROMPT,
};
