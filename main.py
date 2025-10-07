from flask import Flask, request
import src.modules.file_manager as file_manager_module
import src.modules.document_processor as document_processor_module
import src.modules.vector_store_and_embedding as vector_store_and_embedding_module

app = Flask(__name__)

file_manager = file_manager_module.FileManager()
document_processor = document_processor_module.DocumentProcessor()
vse = vector_store_and_embedding_module.VectorStoreAndEmbedding()

@app.route('/uploadFile', methods = ['POST'])
def uploadFile():
    return file_manager.uploadFile()    

@app.route('/uploadMediaFile', methods = ['POST'])
def uploadMediaFile():
    return file_manager.uploadMediaFile()

@app.route('/webFileUpload', methods = ['POST'])
def webFileUpload():
    return file_manager.webFileUpload()

@app.route('/chunkDocument', methods = ['POST'])
def chunkDocument():    
    chunks = document_processor.chunkingDocument()
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    return {'chunks': chunk_texts}

@app.route('/store_chunks', methods = ['POST'])
def store_chunks():
    chunks = document_processor.chunkingDocument()
    chunk_texts = [chunk.page_content for chunk in chunks]
    return vse.store_chunks(chunk_texts)
    



if __name__ == "__main__":
    app.run(debug=True)