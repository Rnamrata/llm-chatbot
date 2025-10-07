from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
import numpy as np
from langchain.schema import Document

class VectorStoreAndEmbedding:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name="my_documents",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
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
    
    def search(self, query, k=5):
        """Search for similar chunks"""
        results = self.vectorstore.similarity_search(query, k=k)
        return results
