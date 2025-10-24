"""
Code Review Prompts Module
==========================
This module contains specialized prompt templates for different types of code review operations.
Each prompt is carefully crafted to elicit specific types of analysis from the LLM.

The prompts use template variables (like {code}, {language}, {context}) that are filled in
at runtime with actual values. This allows the same prompt structure to be reused for
different code review requests.

Available Prompt Types:
- Comprehensive: Full code review covering quality, bugs, security, performance
- Quick: Fast review focusing on critical issues only
- Security: Deep security vulnerability analysis
- Performance: Performance optimization focus
- Best Practices: Design patterns and architectural review
- Conversational: Interactive Q&A about code
- Explanation: Code explanation in plain language
- Improvement: Specific improvement suggestions
- Bug Detection: Finding bugs and potential issues
- Comparison: Reviewing code changes (before/after)
"""

# Import LangChain's PromptTemplate class for creating structured prompts with variables
from langchain.prompts import PromptTemplate


# ==================== COMPREHENSIVE CODE REVIEW PROMPT ====================
# This is the most thorough prompt template, covering all aspects of code quality
#
# Template Variables:
# - {language}: Programming language (e.g., "python", "javascript") - used for syntax highlighting
# - {code}: The actual code to be reviewed
# - {source}: Filename or source identifier (e.g., "main.py")
# - {start_line}/{end_line}: Line number range in the original file
# - {context}: Related code from the codebase retrieved via vector search
# - {chat_history}: Previous conversation messages for continuity
# - {question}: Specific question or focus area from the user
#
# This prompt instructs the LLM to perform a comprehensive multi-dimensional analysis
# covering quality, best practices, bugs, security, performance, and maintainability

COMPREHENSIVE_CODE_REVIEW_PROMPT = """You are an expert code reviewer with years of experience in software engineering best practices, security, and performance optimization.

Review the following code and provide detailed, actionable feedback:

**Code to Review:**
```{language}
{code}
```

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
"""


# Quick code review prompt (faster, less detailed)
QUICK_CODE_REVIEW_PROMPT = """You are a code reviewer. Quickly analyze this code for critical issues:

**Code:**
```{language}
{code}
```

**Source:** {source}

Focus on:
1. Critical bugs or security issues
2. Major performance problems
3. Obvious best practice violations

Provide concise, actionable feedback with line references.

Answer:"""


# Specific aspect review prompts
SECURITY_REVIEW_PROMPT = """You are a security expert. Review this code specifically for security vulnerabilities:

**Code:**
```{language}
{code}
```

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

Answer:"""


PERFORMANCE_REVIEW_PROMPT = """You are a performance optimization expert. Review this code for performance issues:

**Code:**
```{language}
{code}
```

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

Answer:"""


BEST_PRACTICES_PROMPT = """You are a software architecture expert. Review this code for best practices and design patterns:

**Code:**
```{language}
{code}
```

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

Answer:"""


# Conversational code review with context
CONVERSATIONAL_CODE_REVIEW_PROMPT = """You are an experienced code reviewer having a conversation with a developer about their code.

**Code being discussed:**
```{language}
{code}
```

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

Answer:"""


# Code explanation prompt
CODE_EXPLANATION_PROMPT = """You are a senior developer explaining code to help someone understand it better.

**Code:**
```{language}
{code}
```

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

Answer:"""


# Code improvement suggestions
CODE_IMPROVEMENT_PROMPT = """You are a code quality expert. Suggest specific improvements for this code:

**Current Code:**
```{language}
{code}
```

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

Answer:"""


# Bug detection prompt
BUG_DETECTION_PROMPT = """You are a debugging expert. Analyze this code for potential bugs:

**Code:**
```{language}
{code}
```

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

Answer:"""


# Code comparison prompt (for reviewing changes/diffs)
CODE_COMPARISON_PROMPT = """You are reviewing code changes. Compare the before and after:

**Original Code:**
```{language}
{original_code}
```

**Modified Code:**
```{language}
{modified_code}
```

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

Answer:"""


def get_review_prompt_template(review_type: str = "comprehensive") -> PromptTemplate:
    """
    Factory function to get the appropriate prompt template based on review type.

    This function acts as a centralized way to access different code review prompts.
    It maps review type strings to their corresponding prompt templates and returns
    a LangChain PromptTemplate object that can be used with LLMs.

    How it works:
    1. Takes a review_type string (e.g., "security", "performance")
    2. Looks up the corresponding prompt template from the prompts dictionary
    3. If not found, defaults to COMPREHENSIVE_CODE_REVIEW_PROMPT
    4. Wraps the template string in a PromptTemplate object
    5. Specifies the input variables that the template expects

    The PromptTemplate object allows you to fill in variables like:
    >>> template = get_review_prompt_template("security")
    >>> filled_prompt = template.format(code="...", language="python", ...)

    Args:
        review_type (str): The type of review to perform. Options:
            - "comprehensive": Full detailed review (default)
            - "quick": Fast review of critical issues only
            - "security": Security-focused analysis
            - "performance": Performance optimization focus
            - "best_practices": Design patterns and architecture
            - "conversational": Interactive Q&A style
            - "explanation": Code explanation
            - "improvement": Improvement suggestions
            - "bug_detection": Find bugs and issues
            - "comparison": Compare code changes

    Returns:
        PromptTemplate: A LangChain PromptTemplate object configured with:
            - template: The prompt string with {variable} placeholders
            - input_variables: List of variable names used in the template
                              ["code", "language", "source", "context", "chat_history", "question"]

    Example:
        >>> prompt = get_review_prompt_template("security")
        >>> filled = prompt.format(
        ...     code="user_input = request.form['data']",
        ...     language="python",
        ...     source="app.py",
        ...     context="No context",
        ...     chat_history="",
        ...     question="Is this secure?"
        ... )
    """
    # Dictionary mapping review type names to their corresponding prompt templates
    # This makes it easy to add new review types by just adding a new entry
    prompts = {
        "comprehensive": COMPREHENSIVE_CODE_REVIEW_PROMPT,
        "quick": QUICK_CODE_REVIEW_PROMPT,
        "security": SECURITY_REVIEW_PROMPT,
        "performance": PERFORMANCE_REVIEW_PROMPT,
        "best_practices": BEST_PRACTICES_PROMPT,
        "conversational": CONVERSATIONAL_CODE_REVIEW_PROMPT,
        "explanation": CODE_EXPLANATION_PROMPT,
        "improvement": CODE_IMPROVEMENT_PROMPT,
        "bug_detection": BUG_DETECTION_PROMPT,
        "comparison": CODE_COMPARISON_PROMPT,
    }

    # Look up the requested template, defaulting to comprehensive if not found
    # This provides a safe fallback behavior
    template = prompts.get(review_type, COMPREHENSIVE_CODE_REVIEW_PROMPT)

    # Create and return a LangChain PromptTemplate object
    # The input_variables list defines what variables must be provided when using this template
    return PromptTemplate(
        template=template,  # The prompt string with {variable} placeholders
        input_variables=["code", "language", "source", "context", "chat_history", "question"]
    )


def format_code_context(documents, max_context_length=1500):
    """
    Format retrieved code documents into a readable context string for the LLM.

    When performing code review, we often want to provide the LLM with related code
    from the same codebase for better understanding. This function takes Document
    objects retrieved from the vector store and formats them into a nice markdown
    string that can be included in the prompt.

    The function:
    1. Extracts metadata from each document (filename, language, line numbers)
    2. Formats each piece of code in a markdown code block with syntax highlighting
    3. Respects a maximum length limit to avoid overwhelming the LLM
    4. Truncates individual documents if they're too long (keeps first 500 chars)

    Args:
        documents (List[Document]): List of LangChain Document objects retrieved
                                   from the vector store. Each should have:
                                   - page_content: The code text
                                   - metadata: Dict with 'source', 'language', 'start_line', 'end_line'
        max_context_length (int): Maximum total characters for all context.
                                 Default 1500 to keep prompts manageable.
                                 Prevents token limit issues with the LLM.

    Returns:
        str: Formatted markdown string containing code snippets, or a message
             indicating no context is available.

    Example output:
        **From utils.py (lines 10-25):**
        ```python
        def calculate_total(items):
            return sum(item.price for item in items)
        ```

        **From models.py (lines 5-15):**
        ```python
        class Item:
            def __init__(self, price):
                self.price = price
        ```
    """
    # Handle empty document list
    if not documents:
        return "No additional context available."

    # List to collect formatted pieces of context
    context_parts = []

    # Track total length to respect max_context_length limit
    current_length = 0

    # Process each retrieved document
    for doc in documents:
        # Extract metadata (use defaults if keys don't exist)
        source = doc.metadata.get('source', 'unknown')           # Filename
        language = doc.metadata.get('language', 'unknown')       # Programming language
        start_line = doc.metadata.get('start_line', '')          # Starting line number
        end_line = doc.metadata.get('end_line', '')              # Ending line number

        # Format line number info if available (e.g., "(lines 10-25)")
        line_info = f"(lines {start_line}-{end_line})" if start_line else ""

        # Create a formatted markdown string for this code snippet
        # Truncate code to 500 chars to avoid very long snippets
        # Format: **From filename (lines X-Y):**\n```language\ncode\n```
        part = f"\n**From {source} {line_info}:**\n```{language}\n{doc.page_content[:500]}\n```\n"

        # Check if adding this part would exceed our length limit
        if current_length + len(part) > max_context_length:
            # Stop adding more context - we've hit the limit
            break

        # Add this formatted snippet to our list
        context_parts.append(part)
        # Update our running total
        current_length += len(part)

    # If we didn't add any context (maybe all were too long), return message
    if not context_parts:
        return "No additional context available."

    # Join all context parts with newlines and return
    return "\n".join(context_parts)
