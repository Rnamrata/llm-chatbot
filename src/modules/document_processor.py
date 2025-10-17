from flask import request
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader

class DocumentProcessor:

    def __init__(self):
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n\n", "\n\n", "\n", ".", " ", ""]
        )
        self.recursive_splitter = recursive_splitter

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        self.markdown_splitter = markdown_splitter

    def pdfToText(self, file_path, file_name):
        try:
            reader = PdfReader(file_path)
            content = ""
            for page_num, page in enumerate(reader.pages):
                content += page.extract_text() + "\n"
            print(f"PDF: {file_name} - {len(reader.pages)} pages")
            return content
        except Exception as e:
            print(f"Error reading PDF {file_name}: {e}")
            return ""

    def textFileToText(self, file_path, file_name):
        """Extract text from text/markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Text file: {file_name}")
            return content
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            return ""

    def chunkDocument(self, content, metadata):
        """
        Chunk a single document
        
        Args:
            content: Text content to chunk
            metadata: Metadata dictionary for the document
        
        Returns:
            List of Document chunks
        """
        if not content:
            print("No content to chunk")
            return []
        
        # Create Document object
        doc = Document(page_content=content, metadata=metadata)
        
        # Apply markdown splitting
        md_splits = self.markdown_splitter.split_text(doc.page_content)
        
        # Apply recursive splitting
        final_chunks = self.recursive_splitter.split_documents(md_splits)
        
        print(f"Created {len(final_chunks)} chunks")
        return final_chunks