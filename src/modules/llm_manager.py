# src/modules/llm_manager.py
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


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