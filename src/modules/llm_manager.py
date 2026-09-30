# src/modules/llm_manager.py
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from src import config

DOCS_PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following context from documents and the chat history to answer the question.
If you cannot answer based on the context provided, say so clearly.

Context from documents:
{context}

Chat History:
{chat_history}

Current Question: {question}

Helpful Answer:"""

CODE_REVIEW_PROMPT_TEMPLATE = """You are a senior engineer discussing the results of an automated code review with the developer who wrote the code.
Use the following review findings and chat history to answer the question.

For every point you make:
- Cite the specific file and line number the finding refers to
- Explain the reasoning behind the finding, not just what it says
- Suggest a concrete fix when one applies

If the question isn't covered by the findings or code below, say so clearly instead of guessing.

Review findings and code context:
{context}

Chat History:
{chat_history}

Current Question: {question}

Helpful Answer:"""

PROMPT_TEMPLATES = {
    "docs": DOCS_PROMPT_TEMPLATE,
    "code_review": CODE_REVIEW_PROMPT_TEMPLATE,
}

class LLMManager:
    """Manages LLM initialization and query processing"""
    
    def __init__(self, model_name=None, temperature=None):
        """
        Initialize LLM Manager
        
         Args:
            model_name: Name of the Ollama model to use (defaults to config.LLM_MODEL)
            temperature: Temperature for general chat (defaults to config.LLM_TEMPERATURE)
        """
        self.model_name = model_name or config.LLM_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.llm = self._initialize_llm(self.temperature)
        self.review_llm = self._initialize_llm(config.REVIEW_LLM_TEMPERATURE)
        
    def _initialize_llm(self, temperature):
        """Initialize the Ollama LLM"""
        return OllamaLLM(
            model=self.model_name,
            temperature=temperature
        )
    
    def create_conversational_chain(self, retriever, mode="docs"):
        """
        Create a conversational retrieval chain
        
          Args:
            retriever: Vector store retriever
            mode: "docs" for general document Q&A, or "code_review" to discuss
                  review findings (cites file/line, explains reasoning, suggests fixes)

        Returns:
            tuple: (chain, memory)
        """
        if mode not in PROMPT_TEMPLATES:
            raise ValueError(f"Unknown mode: {mode}")
        
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATES[mode],
            input_variables=["context", "chat_history", "question"]
        )

        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        llm = self.review_llm if mode == "code_review" else self.llm

        # Create the conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )
        
        return chain, memory

    def summarize_review(self, findings):
        """
        Turn a list of review findings into a short opening chat reply

        Args:
            findings: List of finding dicts with file/line/severity/message

        Returns:
            str: A short summary, e.g. "3 issues found, 1 high severity..."
        """
        if not findings:
            return "No issues found - the code looks clean!"
        
        findings_text = "\n".join(
            f"-[{f.get('severity', 'info').upper()}] {f.get('file', '?')}:{f.get('line', '?')} - {f.get('message', '')}"
            for f in findings
        )
        prompt = f"""Summarize the following code review findings in 1-2 short sentences,
as an opening chat reply to the developer (e.g. "3 issues found, 1 high severity...").
Do not list every finding individually — just give the headline.

Findings:
{findings_text}

Summary:"""

        try:
            return self.review_llm.invoke(prompt).strip()
        except Exception as e:
            print(f"Error summarizing review findings: {e}")
            return f"Found {len(findings)} issue(s), but couldn't generate summary" 
    
    def simple_query(self, prompt):
        """
        Simple query without retrieval (direct LLM call)
        
        Args:
            prompt: The prompt to send to LLM
        
        Returns:
            str: LLM response
        """
        try:
            response = self.llm.invoke(prompt)
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