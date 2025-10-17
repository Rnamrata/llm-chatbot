from flask import request
import yt_dlp
import os
import sys
from langchain.schema import Document
import whisper
from langchain.document_loaders import WebBaseLoader
from .code_parser import CodeParser


class FileManager:
    def __init__(self, document_processor, vector_store, code_parser=None):
        """
        Initialize FileManager with dependencies

        Args:
            document_processor: DocumentProcessor instance
            vector_store: VectorStoreAndEmbedding instance
            code_parser: CodeParser instance (optional, created if not provided)
        """
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.code_parser = code_parser if code_parser else CodeParser()

    def uploadFile(self):
        """
        Upload file from device (documents or code)
        Automatically: Extract → Chunk → Embed → Store
        Detects file type and routes to appropriate processor
        """
        try:
            uploaded_file = request.files['file']
            filename = uploaded_file.filename
            destination = 'uploads/' + uploaded_file.filename
            uploaded_file.save(destination)

            # Check if it's a code file
            if self.code_parser.is_code_file(filename):
                # Handle as code file
                with open(destination, 'r', encoding='utf-8') as f:
                    code_content = f.read()

                # Use code-specific chunking
                chunks = self.code_parser.chunk_code(code_content, filename)

                if not chunks:
                    return {'error': 'Failed to parse code file', 'success': False}

                # Store in vector database
                store_result = self.vector_store.store_chunks(chunks)

                language = self.code_parser.detect_language(filename)

                return {
                    'success': True,
                    'message': 'Code file uploaded and processed successfully',
                    'filename': filename,
                    'file_type': 'code',
                    'language': language,
                    'chunks_created': store_result['count']
                }

            # Handle as document file
            elif filename.endswith('.pdf'):
                content = self.document_processor.pdfToText(destination, filename)
            elif filename.endswith(('.txt', '.md')):
                content = self.document_processor.textFileToText(destination, filename)
            else:
                return {'error': 'Unsupported file type', 'success': False}

            if not content:
                return {'error': 'No content extracted from file', 'success': False}

            # Chunk the content
            chunks = self.document_processor.chunkDocument(
                content=content,
                metadata={'source': filename, 'type': 'file_upload', 'content_type': 'document'}
            )

            if not chunks:
                return {'error': 'Failed to create chunks', 'success': False}

            # Store in vector database
            store_result = self.vector_store.store_chunks(chunks)

            return {
                'success': True,
                'message': 'File uploaded and processed successfully',
                'filename': filename,
                'file_type': 'document',
                'chunks_created': store_result['count']
            }
        except Exception as e:
            print(f"Error in uploadFile: {e}")
            return {'error': str(e), 'success': False}

    def uploadCodeForReview(self):
        """
        Upload code file specifically for code review
        Provides additional metadata and complexity analysis
        """
        try:
            uploaded_file = request.files['file']
            filename = uploaded_file.filename
            destination = 'uploads/' + uploaded_file.filename
            uploaded_file.save(destination)

            # Verify it's a code file
            if not self.code_parser.is_code_file(filename):
                return {'error': 'File is not a supported code file', 'success': False}

            # Read code content
            with open(destination, 'r', encoding='utf-8') as f:
                code_content = f.read()

            language = self.code_parser.detect_language(filename)

            # Calculate complexity metrics
            complexity = self.code_parser.calculate_complexity(code_content, language)

            # Chunk the code
            chunks = self.code_parser.chunk_code(code_content, filename)

            if not chunks:
                return {'error': 'Failed to parse code file', 'success': False}

            # Store in vector database
            store_result = self.vector_store.store_chunks(chunks)

            return {
                'success': True,
                'message': 'Code file uploaded for review successfully',
                'filename': filename,
                'language': language,
                'chunks_created': store_result['count'],
                'complexity': complexity
            }

        except Exception as e:
            print(f"Error in uploadCodeForReview: {e}")
            return {'error': str(e), 'success': False}

    def uploadCodeDirectory(self):
        """
        Upload multiple code files from a directory (future enhancement)
        For now, returns a placeholder
        """
        return {
            'success': False,
            'error': 'Directory upload not yet implemented. Please upload files individually.'
        }

    def downloadYouTubeFile(self, save_dir, url):
        # Download audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(save_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded audio file
        audio_files = [f for f in os.listdir(save_dir) if f.endswith(('.mp3', '.m4a', '.wav'))]

        if not audio_files:
            return None  # No audio file found   
        return os.path.join(save_dir, audio_files[0])

    def transcribeAudioFile(self, audio_file_path, url):
        whisper_model = whisper.load_model("base")

        # Transcribe
        print(f"Transcribing {audio_file_path}...")
        result = whisper_model.transcribe(audio_file_path)
        
        doc = Document(
            page_content=result["text"],
            metadata={"source": url, "file": os.path.basename(audio_file_path)}
        )

        return doc

    def uploadMediaFile(self):
        """
        Upload YouTube video
        Automatically: Download → Transcribe → Chunk → Embed → Store
        """
        try:
            url = request.form.get('url') if request.form.get('url') else request.json.get('url')
            if not url:
                return {'error': 'No URL provided', 'success': False}
            
            save_dir = 'uploads/media/'
            os.makedirs(save_dir, exist_ok=True)

            # Download audio
            audio_file_path = self.downloadYouTubeFile(save_dir, url)
            
            if not audio_file_path:
                return {'error': 'Failed to download audio from YouTube', 'success': False}
            
            # Transcribe audio
            doc = self.transcribeAudioFile(audio_file_path, url)
            
            # Save transcription to text file
            base_filename = os.path.splitext(doc.metadata['file'])[0]
            transcription_filename = f"{base_filename}_transcription.txt"
            transcription_path = os.path.join('uploads', transcription_filename)
            
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(doc.page_content)

            # Chunk the transcription
            chunks = self.document_processor.chunkDocument(
                content=doc.page_content,
                metadata={'source': url, 'type': 'youtube', 'filename': transcription_filename}
            )
            
            if not chunks:
                return {'error': 'Failed to create chunks', 'success': False}
            
            # Store in vector database
            store_result = self.vector_store.store_chunks(chunks)
            
            return {
                'success': True,
                'message': 'YouTube video processed and stored successfully',
                'url': url,
                'transcription_file': transcription_filename,
                'chunks_created': store_result['count']
            }
            
        except Exception as e:
            print(f"Error in uploadMediaFile: {e}")
            return {'error': str(e), 'success': False}

    def webFileUpload(self):
        """
        Upload web page content
        Automatically: Scrape → Chunk → Embed → Store
        """
        try:
            url = request.form.get('url') if request.form.get('url') else request.json.get('url')
            if not url:
                return {'error': 'No URL provided', 'success': False}
            
            # Load web content
            loader = WebBaseLoader(url)
            data = loader.load()

            if not data:
                return {'error': 'Failed to load data from URL', 'success': False}
            
            doc = data[0]
            
            # Create a safe filename from the title
            title = doc.metadata.get('title', 'web_content')
            safe_filename = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
            safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in ('_', '-', '.'))
            
            # Save the page content to a text file
            destination = f'uploads/{safe_filename}.txt'
            
            with open(destination, 'w', encoding='utf-8') as f:
                f.write(f"Source: {doc.metadata.get('source', 'N/A')}\n")
                f.write(f"Title: {doc.metadata.get('title', 'N/A')}\n")
                f.write(f"Description: {doc.metadata.get('description', 'N/A')}\n")
                f.write(f"Language: {doc.metadata.get('language', 'N/A')}\n")
                f.write("\n" + "="*80 + "\n\n")
                f.write(doc.page_content)

            print(f"Web content saved to: {destination}")
            
            # Chunk the content
            chunks = self.document_processor.chunkDocument(
                content=doc.page_content,
                metadata={'source': url, 'type': 'web', 'title': title, 'filename': safe_filename}
            )
            
            if not chunks:
                return {'error': 'Failed to create chunks', 'success': False}
            
            # Store in vector database
            store_result = self.vector_store.store_chunks(chunks)
            
            return {
                'success': True,
                'message': 'Web page processed and stored successfully',
                'url': url,
                'filename': f'{safe_filename}.txt',
                'chunks_created': store_result['count']
            }
            
        except Exception as e:
            print(f"Error in webFileUpload: {e}")
            return {'error': str(e), 'success': False}