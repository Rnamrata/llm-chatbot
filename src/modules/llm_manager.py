# src/modules/llm_manager.py
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from .code_review_prompts import (
    get_review_prompt_template,
    format_code_context,
    QUICK_CODE_REVIEW_PROMPT,
    CONVERSATIONAL_CODE_REVIEW_PROMPT
)


class LLMManager:
    """Manages LLM initialization and query processing"""
    
    def __init__(self, model_name="llama3.2", temperature=0.7):
        """
        Initialize LLM Manager
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for response generation
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the Ollama LLM"""
        return Ollama(
            model=self.model_name,
            temperature=self.temperature
        )
    
    def create_conversational_chain(self, retriever, k=5):
        """
        Create a conversational retrieval chain
        
        Args:
            retriever: Vector store retriever
            k: Number of documents to retrieve
        
        Returns:
            tuple: (chain, memory)
        """
        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create the conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        return chain, memory
    
    def create_custom_prompt_chain(self, retriever, k=5):
        """
        Create a conversational chain with custom prompt
        
        Args:
            retriever: Vector store retriever
            k: Number of documents to retrieve
        
        Returns:
            tuple: (chain, memory)
        """
        # Custom prompt template
        prompt_template = """You are a helpful AI assistant. Use the following context from documents and the chat history to answer the question.
        If you cannot answer based on the context provided, say so clearly.

        Context from documents:
        {context}

        Chat History:
        {chat_history}

        Current Question: {question}

        Helpful Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create chain with custom prompt
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )
        
        return chain, memory
    
    def simple_query(self, prompt):
        """
        Simple query without retrieval (direct LLM call)
        
        Args:
            prompt: The prompt to send to LLM
        
        Returns:
            str: LLM response
        """
        try:
            response = self.llm(prompt)
            return response
        except Exception as e:
            print(f"Error in simple query: {e}")
            return f"Error: {str(e)}"
    
    def format_sources(self, source_documents, max_length=200):
        """
        Format source documents for response

        Args:
            source_documents: List of retrieved documents
            max_length: Maximum length for content preview

        Returns:
            list: Formatted source information
        """
        sources = []
        for doc in source_documents:
            content = doc.page_content
            preview = content[:max_length] + "..." if len(content) > max_length else content

            source_info = {
                'content': preview,
                'metadata': doc.metadata,
                'full_length': len(content)
            }
            sources.append(source_info)

        return sources

    def create_code_review_chain(self, retriever, review_type="conversational", k=5):
        """
        Create a code review chain with specialized prompts

        Args:
            retriever: Vector store retriever for code context
            review_type: Type of review (comprehensive, quick, security, etc.)
            k: Number of code chunks to retrieve

        Returns:
            tuple: (chain, memory)
        """
        # Get appropriate prompt template
        prompt = get_review_prompt_template(review_type)

        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        # Create the conversational chain with code review prompt
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )

        return chain, memory

    def review_code_direct(self, code, language, source, review_type="quick"):
        """
        Review code directly without RAG retrieval

        Args:
            code: Code content to review
            language: Programming language
            source: Source filename
            review_type: Type of review to perform

        Returns:
            str: Review feedback
        """
        try:
            # Get the appropriate prompt template
            prompt_template = get_review_prompt_template(review_type)

            # Format the prompt with code information
            prompt = prompt_template.format(
                code=code,
                language=language,
                source=source,
                context="No additional context available (direct review)",
                chat_history="",
                question="Please review this code.",
                start_line="",
                end_line=""
            )

            # Get LLM response
            response = self.llm(prompt)
            return response

        except Exception as e:
            print(f"Error in review_code_direct: {e}")
            return f"Error performing code review: {str(e)}"

    def review_code_with_context(self, code, language, source,
                                  context_documents, question="Please review this code.",
                                  review_type="comprehensive"):
        """
        Review code with context from related code in the codebase

        Args:
            code: Code content to review
            language: Programming language
            source: Source filename
            context_documents: Related code documents from vector store
            question: Specific question or review focus
            review_type: Type of review to perform

        Returns:
            str: Review feedback
        """
        try:
            # Format context from documents
            context = format_code_context(context_documents)

            # Get the appropriate prompt template
            prompt_template = get_review_prompt_template(review_type)

            # Format the prompt
            prompt = prompt_template.format(
                code=code,
                language=language,
                source=source,
                context=context,
                chat_history="",
                question=question,
                start_line="",
                end_line=""
            )

            # Get LLM response
            response = self.llm(prompt)
            return response

        except Exception as e:
            print(f"Error in review_code_with_context: {e}")
            return f"Error performing code review: {str(e)}"

    def explain_code(self, code, language, source, context_documents=None,
                     question="What does this code do?"):
        """
        Explain what code does in clear language

        Args:
            code: Code content to explain
            language: Programming language
            source: Source filename
            context_documents: Optional related code for context
            question: Specific question about the code

        Returns:
            str: Code explanation
        """
        try:
            context = format_code_context(context_documents) if context_documents else "No additional context"

            prompt_template = get_review_prompt_template("explanation")

            prompt = prompt_template.format(
                code=code,
                language=language,
                source=source,
                context=context,
                chat_history="",
                question=question
            )

            response = self.llm(prompt)
            return response

        except Exception as e:
            print(f"Error in explain_code: {e}")
            return f"Error explaining code: {str(e)}"

    def suggest_improvements(self, code, language, source, goal="",
                            context_documents=None):
        """
        Suggest specific improvements for code

        Args:
            code: Code content to improve
            language: Programming language
            source: Source filename
            goal: Developer's improvement goal
            context_documents: Optional related code for context

        Returns:
            str: Improvement suggestions
        """
        try:
            context = format_code_context(context_documents) if context_documents else "No additional context"

            prompt_template = get_review_prompt_template("improvement")

            question = goal if goal else "How can I improve this code?"

            prompt = prompt_template.format(
                code=code,
                language=language,
                source=source,
                context=context,
                chat_history="",
                question=question
            )

            response = self.llm(prompt)
            return response

        except Exception as e:
            print(f"Error in suggest_improvements: {e}")
            return f"Error suggesting improvements: {str(e)}"

    def detect_bugs(self, code, language, source, reported_issue="",
                    context_documents=None):
        """
        Detect potential bugs in code

        Args:
            code: Code content to analyze
            language: Programming language
            source: Source filename
            reported_issue: Reported bug or issue (if any)
            context_documents: Optional related code for context

        Returns:
            str: Bug analysis
        """
        try:
            context = format_code_context(context_documents) if context_documents else "No additional context"

            prompt_template = get_review_prompt_template("bug_detection")

            question = reported_issue if reported_issue else "Are there any bugs in this code?"

            prompt = prompt_template.format(
                code=code,
                language=language,
                source=source,
                context=context,
                chat_history="",
                question=question
            )

            response = self.llm(prompt)
            return response

        except Exception as e:
            print(f"Error in detect_bugs: {e}")
            return f"Error detecting bugs: {str(e)}"