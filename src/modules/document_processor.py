import os
from src import config
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
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

        # Language-specific splitters, created lazily and cached on first use
        self.code_splitters = {}

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

    def codeFileToText(self, file_path, file_name):
        """
        Extract text from a source code file

        Args:
            file_path: Path to the code file on disk
            file_name: Original filename, used to look up its language

        Returns:
            tuple: (content, language, line_count). Language is None and
            content is "" if the file can't be read.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            ext = os.path.splitext(file_name)[1].lower()
            language = config.CODE_LANGUAGES.get(ext, None)
            line_count = content.count('\n') + 1 if content else 0

            print(f"Code files: {file_name} - {language} - {line_count} lines")
            return content, language, line_count
        except Exception as e:
            print(f"Error reading code file {file_name}: {e}")
            return "", None, 0

    def _get_code_splitter(self, language):
        """Return the cached splitter for this language, creating it on first use"""
        if language not in self.code_splitters:
            self.code_splitters[language] = RecursiveCharacterTextSplitter.from_language(
                language=Language(language),
                chunk_size=config.CODE_CHUNK_SIZE,
                chunk_overlap=config.CODE_CHUNK_OVERLAP
            )

        return self.code_splitters[language]

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

        source = metadata.get("source", "")

        if source.endswith(".md"):
            # MarkdownHeaderTextSplitter only keeps header info in metadata,
            # so re-attach the original metadata (source, type, ...) afterward
            md_splits = self.markdown_splitter.split_text(content)
            for split in md_splits:
                split.metadata.update(metadata)
            final_chunks = self.recursive_splitter.split_documents(md_splits)
        else:
            # Create Document object
            doc = Document(page_content=content, metadata=metadata)
            final_chunks = self.recursive_splitter.split_documents([doc])
        
        print(f"Created {len(final_chunks)} chunks")
        return final_chunks

    def chunkCode(self, content, language, metadata):
        """
        Chunk source code using a language-aware splitter that tries to
        break at function/class boundaries instead of arbitrary character counts

        Args:
            content: Source code to chunk
            language: Language identifier matching LangChain's Language enum
                    (e.g. "python", "js", "java") — see config.CODE_LANGUAGES
            metadata: Metadata dictionary for the document

        Returns:
            List of Document chunks, each with start_line/end_line metadata
        """

        if not content:
            print("No code content to chunk")
            return []

        splitter = self._get_code_splitter(language)
        raw_chunks = splitter.split_text(content)

        final_chunk = []
        search_from = 0
        for chunk_text in raw_chunks:
            start_index = content.find(chunk_text, search_from)
            if start_index == -1:
                start_index = content.find(chunk_text)

            if start_index != -1:
                search_from = start_index
                start_line = content.count('\n', 0, start_index) + 1
                end_line = start_line + chunk_text.count('\n')
            else:
                start_line = end_line = None

            chunk_metadata = {**metadata, "start_line": start_line, "end_line": end_line}
            final_chunk.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        print(f"Created {len(final_chunk)} code chunks")
        return final_chunk
    