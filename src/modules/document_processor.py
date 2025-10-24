# Import Flask's request object to handle file uploads from HTTP requests
from flask import request
# Import os module for file system operations
import os
# Import LangChain text splitters for intelligent document chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
# Import LangChain Document class to create structured documents
from langchain.schema import Document
# Import PyPDF2 for reading PDF files
from PyPDF2 import PdfReader

class DocumentProcessor:
    """
    Document Processing and Chunking Module
    ========================================
    This class handles the extraction and intelligent chunking of various document types
    for storage in a vector database. It supports PDFs, text files, and markdown files.

    Key Features:
    - PDF text extraction (page by page)
    - Text/Markdown file reading
    - Intelligent chunking that respects document structure
    - Markdown-aware splitting (preserves headers)
    - Recursive character splitting with overlap for context preservation

    The two-stage chunking process:
    1. Markdown splitting: First splits on markdown headers (# and ##) to respect structure
    2. Recursive splitting: Then splits large sections into smaller chunks with overlap

    Chunk overlap is important for RAG systems because it ensures context isn't lost
    at chunk boundaries. For example, if a sentence is split across chunks, the overlap
    ensures both chunks contain the complete sentence.
    """

    def __init__(self):
        """
        Initialize the DocumentProcessor with configured text splitters.

        Sets up two complementary text splitters:
        1. RecursiveCharacterTextSplitter: Splits text intelligently by trying multiple separators
        2. MarkdownHeaderTextSplitter: Preserves markdown document structure
        """
        # Create a recursive splitter that tries multiple separators in order
        # This ensures text is split at natural boundaries (paragraphs, sentences, words)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,         # Target chunk size in characters (~250 tokens)
            chunk_overlap=100,       # Overlap between chunks to preserve context (10% overlap)
            # Separators are tried in order - prefers larger natural boundaries
            separators=[
                "\n\n\n",            # Try triple newline first (major sections)
                "\n\n",              # Then double newline (paragraphs)
                "\n",                # Then single newline (lines)
                ".",                 # Then periods (sentences)
                " ",                 # Then spaces (words)
                ""                   # Finally, character-by-character (last resort)
            ]
        )
        self.recursive_splitter = recursive_splitter

        # Configure markdown splitting to preserve document structure
        # Splits on header markers and preserves them in metadata
        headers_to_split_on = [
            ("#", "Header 1"),       # Top-level headers (# Title)
            ("##", "Header 2"),      # Second-level headers (## Subtitle)
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        self.markdown_splitter = markdown_splitter

    def pdfToText(self, file_path, file_name):
        """
        Extract all text content from a PDF file.

        This method reads a PDF file page by page and extracts all text content.
        It uses PyPDF2's text extraction capabilities which work well for text-based PDFs
        but may not work for scanned PDFs (which would need OCR).

        Process:
        1. Create a PdfReader object from the file path
        2. Iterate through all pages in the PDF
        3. Extract text from each page using extract_text()
        4. Concatenate all page text with newlines

        Args:
            file_path (str): Absolute path to the PDF file on disk
            file_name (str): Name of the file (used for logging)

        Returns:
            str: All extracted text from the PDF, or empty string if error occurs

        Note: Scanned PDFs (images of text) will return empty or garbage text
              and would require OCR (Optical Character Recognition) tools.
        """
        try:
            # Create a PDF reader object that can parse the PDF file
            reader = PdfReader(file_path)

            # Initialize empty string to accumulate text from all pages
            content = ""

            # Iterate through each page (enumerate gives us page number too)
            for page_num, page in enumerate(reader.pages):
                # Extract text from this page and add it to our content string
                # Add newline after each page for separation
                content += page.extract_text() + "\n"

            # Log successful extraction with page count
            print(f"PDF: {file_name} - {len(reader.pages)} pages")
            return content

        except Exception as e:
            # If anything goes wrong (file not found, corrupted PDF, etc.), log and return empty
            print(f"Error reading PDF {file_name}: {e}")
            return ""

    def textFileToText(self, file_path, file_name):
        """
        Extract text from plain text or markdown files.

        Simple file reading with UTF-8 encoding to support international characters.
        Works for .txt, .md, and other text-based formats.

        Args:
            file_path (str): Absolute path to the text file
            file_name (str): Name of the file (used for logging)

        Returns:
            str: Complete file contents, or empty string if error occurs
        """
        try:
            # Open file with UTF-8 encoding (supports international characters)
            # 'r' mode = read mode (not writing)
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read entire file content into memory
                content = f.read()

            # Log successful read
            print(f"Text file: {file_name}")
            return content

        except Exception as e:
            # Handle errors (file not found, encoding issues, permission denied, etc.)
            print(f"Error reading {file_name}: {e}")
            return ""

    def chunkDocument(self, content, metadata):
        """
        Chunk a single document using two-stage intelligent splitting.

        This is the core chunking method that applies both markdown-aware and
        recursive splitting to create optimal chunks for RAG systems.

        Two-Stage Process:
        1. Markdown Splitting: First splits on markdown headers (# and ##)
           - Preserves document structure
           - Keeps sections together
           - Stores header information in metadata

        2. Recursive Splitting: Then splits large sections into smaller chunks
           - Ensures chunks don't exceed size limits
           - Uses intelligent separators (paragraphs, sentences, words)
           - Adds overlap between chunks to preserve context

        Why two stages?
        - Markdown splitting preserves logical structure
        - Recursive splitting ensures size constraints
        - Together they create semantically meaningful, appropriately sized chunks

        Args:
            content (str): The document text to chunk
            metadata (dict): Metadata to attach to chunks (source, type, etc.)

        Returns:
            List[Document]: List of Document objects, each containing:
                - page_content: Chunk text
                - metadata: Original metadata plus any added by splitters

        Example:
            >>> processor = DocumentProcessor()
            >>> chunks = processor.chunkDocument(
            ...     "# Introduction\n\nThis is a long document...",
            ...     {"source": "doc.md", "type": "documentation"}
            ... )
            >>> len(chunks)
            5
        """
        # Validate that we have content to process
        if not content:
            print("No content to chunk")
            return []

        # Create a LangChain Document object with the content and metadata
        # This provides a standardized format for the splitters to work with
        doc = Document(page_content=content, metadata=metadata)

        # STAGE 1: Apply markdown-aware splitting
        # This splits on markdown headers (# and ##) to preserve document structure
        # Returns a list of Document objects, one per section
        md_splits = self.markdown_splitter.split_text(doc.page_content)

        # STAGE 2: Apply recursive character splitting
        # This takes the markdown splits and further divides them if they're too large
        # Uses intelligent separators and adds overlap for context preservation
        # Input: List of Documents from markdown splitting
        # Output: List of final chunk Documents (more chunks if sections were too large)
        final_chunks = self.recursive_splitter.split_documents(md_splits)

        # Log the number of final chunks created
        print(f"Created {len(final_chunks)} chunks")
        return final_chunks