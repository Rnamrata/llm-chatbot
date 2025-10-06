from flask import request
import yt_dlp
import os
import sys
from langchain.schema import Document
import whisper
from langchain.document_loaders import WebBaseLoader


class FileManager:

    def uploadFile(self):
        uploaded_file = request.files['file']
        destination = 'uploads/' + uploaded_file.filename
        uploaded_file.save(destination)

        return {'data': 'File Uploaded Successfully'}
    
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
        url = request.form.get('url')  # Fixed: added .get()
        if not url:
            return {'error': 'No URL provided'}, 400
        
        save_dir = 'uploads/media/'
        os.makedirs(save_dir, exist_ok=True)

        audio_file_path = self.downloadYouTubeFile(save_dir, url)
        
        if audio_file_path:
            doc = self.transcribeAudioFile(audio_file_path, url)
            # Save transcription to text file
            base_filename = os.path.splitext(doc.metadata['file'])[0]  # Remove extension
            transcription_filename = f"{base_filename}_transcription.txt"
            transcription_path = os.path.join('uploads', transcription_filename)
            
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(doc.page_content)
            
            return {'data': 'Media File Uploaded and Transcribed Successfully'}
        else:
            return {'error': 'Failed to download audio from the provided URL'}, 400 

    def webFileUpload(self):
        url = request.form.get('url')  # Fixed: added .get()
        if not url:
            return {'error': 'No URL provided'}, 400
        loader = WebBaseLoader(url)
        data = loader.load()

        if not data:
            return {'error': 'Failed to load data from the provided URL'}, 400
        # Get the first document from the list
        doc = data[0]
        
        # Create a safe filename from the title
        title = doc.metadata.get('title', 'web_content')
        safe_filename = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Remove any other problematic characters
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in ('_', '-', '.'))
        
        # Save the page content to a text file
        destination = f'uploads/{safe_filename}.txt'
        
        with open(destination, 'w', encoding='utf-8') as f:
            # Write metadata first
            f.write(f"Source: {doc.metadata.get('source', 'N/A')}\n")
            f.write(f"Title: {doc.metadata.get('title', 'N/A')}\n")
            f.write(f"Description: {doc.metadata.get('description', 'N/A')}\n")
            f.write(f"Language: {doc.metadata.get('language', 'N/A')}\n")
            f.write("\n" + "="*80 + "\n\n")
            # Write the main content
            f.write(doc.page_content)
        
        print(f"Web content saved to: {destination}")
        
        return {'data': 'Web File Uploaded Successfully'}
