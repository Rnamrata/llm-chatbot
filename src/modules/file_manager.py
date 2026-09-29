from src import config
import yt_dlp
import os
import uuid
from langchain_core.documents import Document
import whisper
from langchain_community.document_loaders import WebBaseLoader
from werkzeug.utils import secure_filename

class FileManager:
    def __init__(self, document_processor, vector_store, review_client):
        """
        Initialize FileManager with dependencies
        
        Args:
            document_processor: DocumentProcessor instance
            vector_store: VectorStoreAndEmbedding instance
        """
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.review_client = review_client

        os.makedirs("uploads", exist_ok=True)
        os.makedirs("uploads/media", exist_ok=True)

        self.whisper_model = whisper.load_model("base")

    def uploadFile(self, file):
        """
        Upload file from device
        Automatically: Extract → Chunk → Embed → Store

        Args:
            file: A file object with .filename and .save(path), e.g. Flask's
                  request.files['file']
        """
        try:
            filename = secure_filename(file.filename)
            destination = os.path.join('uploads', filename)
            file.save(destination)

            # Extract content based on file type
            if filename.endswith('.pdf'):
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
                metadata={'source': filename, 'type': 'file_upload'}
            )
            
            if not chunks:
                return {'error': 'Failed to create chunks', 'success': False}
            
            # Store in vector database
            store_result = self.vector_store.store_chunks(chunks)
            
            return {
                'success': True,
                'message': 'File uploaded and processed successfully',
                'filename': filename,
                'chunks_created': store_result['count']
            }
        except Exception as e:
            print(f"Error in uploadFile: {e}")
            return {'error': str(e), 'success': False}
    
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
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)

        # Find the downloaded audio file
        audio_path = os.path.splitext(downloaded_path)[0] + '.mp3'

        if not os.path.exists(audio_path):
            return None  # No audio file found   
        return audio_path

    def transcribeAudioFile(self, audio_file_path, url):
        # Transcribe
        print(f"Transcribing {audio_file_path}...")
        result = self.whisper_model.transcribe(audio_file_path)
        
        doc = Document(
            page_content=result["text"],
            metadata={"source": url, "file": os.path.basename(audio_file_path)}
        )

        return doc

    def uploadMediaFile(self, url):
        """
        Upload YouTube video
        Automatically: Download → Transcribe → Chunk → Embed → Store

        Args:
            url: YouTube video URL
        """
        try:
            if not url:
                return {'error': 'No URL provided', 'success': False}
            
            save_dir = 'uploads/media/'

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

    def webFileUpload(self, url):
        """
        Upload web page content
        Automatically: Scrape → Chunk → Embed → Store

        Args:
            url: Web page URL
        """
        try:
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

    def uploadCodeForReview(self, file, session_id):
        """
    Upload a code file for automated review
    Automatically: Validate → Save → Review (Node) → Chunk → Embed → Store

    Args:
        file: A file object with .filename and .save(path), e.g. Flask's
              request.files['file']
        session_id: Chat session this review belongs to

    Returns:
        dict: {
            'success': True,
            'review_id': str,
            'findings': [...],
            'code': str,
            'filename': str,
            'language': str,
            'warning': str (only present if the review service was unreachable)
        }
    """
        try:
            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext not in config.CODE_EXTENSIONS:
                return {'error': f'Unsupported code file type: {ext}', 'success': False}

            destination = os.path.join('uploads', filename)
            file.save(destination)

            content, language, line_count = self.document_processor.codeFileToText(destination, filename)

            if not content:
                return {'error': 'No content extracted from file', 'success': False}

            review_id = str(uuid.uuid4())

            # Ask the Node review service for findings — if it's down, keep
            # going so the code still gets stored and is chattable
            review_result = self.review_client.review(filename, language, content)

            warning = None
            findings = []
            if review_result['success']:
                findings = review_result['findings']
            else:
                warning = (
                    f"Review service unavailable ({review_result['error']}); "
                    "code was stored so you can still chat about it"
                )

                print(f"uploadCodeForReview: {warning}")

            # Chunk and store the code itself, scoped to this session/review
            code_metadata = {
                'source': filename,
                'type': 'code_review',
                'language': language,
                'review_id': review_id,
                'session_id': session_id
            }
            code_chunks = self.document_processor.chunkCode(content, language, code_metadata)

            if code_chunks:
                self.vector_store.store_chunks(code_chunks)

            # Store each finding as its own small, retrievable document
            finding_docs = []
            for finding in findings:
                finding_docs.append(Document(
                    page_content=finding.get('message', ''),
                    metadata={
                        'source': filename,
                        'type': 'review_finding',
                        'session_id': session_id,
                        'review_id': review_id,
                        'file': finding.get('file', filename),
                        'line': finding.get('line') or 0,
                        'severity': finding.get('severity', 'info'),
                        'message': finding.get('message', '')
                    }
                ))

            if finding_docs:
                self.vector_store.store_chunks(finding_docs)

            result = {
                'success': True,
                'review_id': review_id,
                'findings': findings,
                'code': content,
                'filename': filename,
                'language': language
            }

            if warning:
                result['warning'] = warning

            return result
        except Exception as e:
            print(f"Error in uploadCodeForReview: {e}")
            return {'error': str(e), 'success': False}