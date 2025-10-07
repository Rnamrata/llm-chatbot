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

    def chunkingDocument(self):
        text_files = os.listdir('uploads/')
        documents = []
        
        for file_name in text_files:
            if file_name.endswith(('.txt', '.pdf', '.md')):
                file_path = os.path.join('uploads/', file_name)
                try:
                    if file_name.endswith('.pdf'):
                        content = self.pdfToText(file_path, file_name)
                    else:
                        # Handle text and markdown files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        print(f"Text file: {file_name}")
                    
                    # Create Document objects
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_name}
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
                    continue
        
        if not documents:
            print("No documents found to chunk")
            return []
        
        # Process each document through markdown splitter, then recursive splitter
        md_splits = []
        for doc in documents:
            splits = self.markdown_splitter.split_text(doc.page_content)
            md_splits.extend(splits)
        
        final_splits = self.recursive_splitter.split_documents(md_splits)
        
        print(f"Total documents loaded: {len(documents)}")
        print(f"After markdown splitting: {len(md_splits)}")
        print(f"Final chunks created: {len(final_splits)}")
        
        return final_splits