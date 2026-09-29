from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src import config

class VectorStoreAndEmbedding:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )

        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(config.CHROMA_PERSIST_DIR)
        )

    def store_chunks(self, chunks):
        """Add chunks to vector store"""
        if not chunks:
            return {'error': 'No chunks provided'}
        
        # Convert strings to Documents if needed
        documents = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                # If it's a string, convert to Document
                doc = Document(
                    page_content=chunk,
                    metadata={"chunk_index": i}
                )
                documents.append(doc)
            elif isinstance(chunk, Document):
                # Already a Document
                documents.append(chunk)
            else:
                print(f"Warning: Skipping invalid chunk type: {type(chunk)}")
        
        # Add to vector store
        self.vectorstore.add_documents(documents)
        print(f"Stored {len(documents)} chunks in vector database")
        return {
            'data': 'Chunks embedded and stored successfully',
            'count': len(documents)
        }
    
    def search(self, query, k=5, filter=None):
        """
        Search for similar chunks

        Args:
            query: Search text
            k: Number of results to return
            filter: Optional Chroma metadata filter (e.g. {"review_id": "abc123"})
                    to scope the search to a specific review/session's chunks
        """
        results = self.vectorstore.max_marginal_relevance_search(query, k=k, filter=filter)
        return results

    def get_retriever(self, k=None, filter=None):
        """
        Build a retriever, optionally scoped to a metadata filter

        Args:
            k: Number of documents to retrieve (defaults to config.RETRIEVAL_K)
            filter: Optional Chroma metadata filter (e.g. {"session_id": "abc123"})
                    so a review session only retrieves its own code/findings,
                    not unrelated documents in the same collection

        Returns:
            A LangChain retriever
        """
        search_kwargs = {"k": k or config.RETRIEVAL_K}
        if filter:
            search_kwargs["filter"] = filter
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def delete_by(self, filter):
        """
        Delete chunks matching a metadata filter

        Args:
            filter: Chroma "where" metadata filter, e.g. {"review_id": "abc123"}
                    or {"session_id": "abc123"}

        Returns:
            dict: {'success': True, 'filter': filter} or {'error': str}
        """

        if not filter:
            print("Error: No filter provided, skipping delete")
            return {'error': 'No filter provided'}

        try:
            self.vectorstore.delete(where=filter)
            print(f"Deleted chunks matching filter: {filter}")
            return {'success': True, 'filter': filter}
        except Exception as e:
            print(f"Error deleting chunks with filter {filter}: {e}")
            return {'error': str(e)}