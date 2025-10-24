# Import Ollama embeddings for local embedding generation (runs on your machine, no API needed)
from langchain_community.embeddings import OllamaEmbeddings
# Import Chroma vector database for storing and searching embeddings
from langchain.vectorstores import Chroma
# Import numpy for numerical operations (not currently used but imported for future use)
import numpy as np
# Import Document class for structured document handling
from langchain.schema import Document

class VectorStoreAndEmbedding:
    """
    Vector Store and Embedding Manager
    ===================================
    Manages the vector database (Chroma) and embedding generation (Ollama) for RAG system.

    This class is responsible for:
    - Converting text chunks into vector embeddings (numerical representations)
    - Storing embeddings in a persistent vector database
    - Searching for similar chunks using semantic similarity

    How Vector Search Works:
    1. Text is converted to embeddings (arrays of numbers that capture meaning)
    2. Similar texts have similar embeddings (close in vector space)
    3. When user asks a question, we embed it and find nearest neighbor chunks
    4. These similar chunks are retrieved as context for the LLM

    The Ollama "nomic-embed-text" model generates 768-dimensional embeddings locally.
    Chroma stores these embeddings and provides fast similarity search.
    """

    def __init__(self):
        """
        Initialize the embedding model and vector database.

        Sets up:
        1. Ollama embeddings model (runs locally, no API key needed)
        2. Chroma vector database (persistent storage on disk)

        The database persists to ./chroma_db directory, so embeddings survive restarts.
        """
        # Initialize the embedding model using Ollama
        # "nomic-embed-text" is optimized for text embedding and runs locally
        # This model converts text strings into 768-dimensional vectors
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"  # Local embedding model (no API calls)
        )

        # Initialize Chroma vector database
        # Chroma stores embeddings and enables fast similarity search
        self.vectorstore = Chroma(
            collection_name="my_documents",              # Name for this collection of embeddings
            embedding_function=self.embeddings,          # Use Ollama for embedding
            persist_directory="./chroma_db"              # Save to disk (persists across restarts)
        )

    def store_chunks(self, chunks):
        """
        Embed and store document chunks in the vector database.

        This method takes text chunks and:
        1. Converts them to Document objects if needed
        2. Generates embeddings for each chunk (using Ollama)
        3. Stores embeddings + text + metadata in Chroma database

        The embeddings allow semantic search later when users ask questions.

        Args:
            chunks (list): List of chunks to store. Can be:
                          - Document objects (with page_content and metadata)
                          - Plain strings (will be converted to Documents)

        Returns:
            dict: Result dictionary with:
                - 'data': Success message
                - 'count': Number of chunks stored
                OR
                - 'error': Error message if no chunks provided
        """
        # Validate input
        if not chunks:
            return {'error': 'No chunks provided'}

        # Convert all chunks to Document objects (standardized format)
        documents = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                # Plain string - convert to Document object
                # Add basic metadata with chunk index
                doc = Document(
                    page_content=chunk,          # The actual text
                    metadata={"chunk_index": i}  # Metadata for tracking
                )
                documents.append(doc)

            elif isinstance(chunk, Document):
                # Already a Document object - use as-is
                # These typically come from DocumentProcessor or CodeParser with rich metadata
                documents.append(chunk)

            else:
                # Unknown type - skip it and warn
                print(f"Warning: Skipping invalid chunk type: {type(chunk)}")

        # Add all documents to the vector store
        # This internally:
        # 1. Calls self.embeddings to generate vector for each chunk
        # 2. Stores embedding + text + metadata in Chroma
        # 3. Persists to disk (./chroma_db)
        self.vectorstore.add_documents(documents)

        # Print statistics
        print(self.vectorstore._collection.count())              # Total items in database
        print(f"Stored {len(documents)} chunks in vector database")

        # Return success response
        return {
            'data': 'Chunks embedded and stored successfully',
            'count': len(documents)
        }

    def search(self, query, k=5):
        """
        Search for semantically similar chunks using Maximum Marginal Relevance.

        MMR (Maximum Marginal Relevance) balances:
        - Similarity to the query (relevance)
        - Diversity among results (avoids redundant similar chunks)

        This is better than pure similarity search because it returns diverse
        relevant chunks rather than k very similar chunks.

        Process:
        1. Convert query to embedding using Ollama
        2. Find chunks with similar embeddings (cosine similarity)
        3. Apply MMR to select diverse relevant chunks
        4. Return top k chunks

        Args:
            query (str): The search query (question or text to find similar chunks for)
            k (int): Number of chunks to return (default: 5)

        Returns:
            list[Document]: List of k most relevant Document objects, each containing:
                - page_content: The chunk text
                - metadata: Associated metadata (source, line numbers, etc.)
        """
        # Perform MMR search in the vector database
        # Returns k Document objects ordered by relevance
        results = self.vectorstore.max_marginal_relevance_search(query, k=k)
        return results
