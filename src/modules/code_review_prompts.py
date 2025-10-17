"""
Code Review Prompts Module
Contains specialized prompts for code review functionality.
"""

from langchain.prompts import PromptTemplate


# Comprehensive code review prompt
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
    Get a prompt template for specific review type.

    Args:
        review_type: Type of review (comprehensive, quick, security, performance,
                     best_practices, conversational, explanation, improvement,
                     bug_detection, comparison)

    Returns:
        PromptTemplate object
    """
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

    template = prompts.get(review_type, COMPREHENSIVE_CODE_REVIEW_PROMPT)

    return PromptTemplate(
        template=template,
        input_variables=["code", "language", "source", "context", "chat_history", "question"]
    )


def format_code_context(documents, max_context_length=1500):
    """
    Format retrieved documents as context for code review.

    Args:
        documents: List of Document objects from vector store
        max_context_length: Maximum characters for context

    Returns:
        Formatted context string
    """
    if not documents:
        return "No additional context available."

    context_parts = []
    current_length = 0

    for doc in documents:
        # Format each document
        source = doc.metadata.get('source', 'unknown')
        language = doc.metadata.get('language', 'unknown')
        start_line = doc.metadata.get('start_line', '')
        end_line = doc.metadata.get('end_line', '')

        line_info = f"(lines {start_line}-{end_line})" if start_line else ""

        part = f"\n**From {source} {line_info}:**\n```{language}\n{doc.page_content[:500]}\n```\n"

        if current_length + len(part) > max_context_length:
            break

        context_parts.append(part)
        current_length += len(part)

    if not context_parts:
        return "No additional context available."

    return "\n".join(context_parts)
