import time
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src import config

# Seconds to wait before retrying a batch; doubles with each retry
RETRY_DELAY_SECONDS = 1.0


class PrefixedEmbeddings(Embeddings):
    """
    Adds the task prefixes nomic-embed-text expects: "search_document: " before each
    stored passage and "search_query: " before each question. Only the vectors see
    them; the stored text stays as it was.
    """

    def __init__(self, base, document_prefix='', query_prefix=''):
        self.base = base
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts):
        return self.base.embed_documents([self.document_prefix + t for t in texts])

    def embed_query(self, text):
        return self.base.embed_query(self.query_prefix + text)


class VectorStoreAndEmbedding:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = PrefixedEmbeddings(
            OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL),
            document_prefix=config.EMBED_DOCUMENT_PREFIX,
            query_prefix=config.EMBED_QUERY_PREFIX,
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
        
        # Embed and store in batches, retrying a batch that fails. All or nothing: if a
        # batch keeps failing, the batches already stored are removed again, so a file
        # is never left half searchable.
        stored_ids = []
        try:
            for start in range(0, len(documents), config.EMBED_BATCH_SIZE):
                batch = documents[start:start + config.EMBED_BATCH_SIZE]
                stored_ids.extend(self._add_with_retry(batch) or [])
        except Exception:
            if stored_ids:
                try:
                    self.vectorstore.delete(ids=stored_ids)
                except Exception as cleanup_error:
                    print(f"⚠️  Could not remove {len(stored_ids)} chunks of a failed upload: {cleanup_error}")
            raise

        print(f"Stored {len(documents)} chunks in vector database")
        return {
            'data': 'Chunks embedded and stored successfully',
            'count': len(documents)
        }

    def _add_with_retry(self, batch):
        """
        Embeds and stores one batch. Ollama sometimes drops a connection mid-request
        ("EOF"), so a failed batch is tried again before giving up.

        Returns:
            list: The ids Chroma gave the stored chunks
        """
        for attempt in range(1, config.EMBED_ATTEMPTS + 1):
            try:
                return self.vectorstore.add_documents(batch)
            except Exception as e:
                if attempt == config.EMBED_ATTEMPTS:
                    raise
                print(f"⚠️  Embedding {len(batch)} chunks failed ({e}); trying again ({attempt}/{config.EMBED_ATTEMPTS - 1})")
                time.sleep(RETRY_DELAY_SECONDS * 2 ** (attempt - 1))

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