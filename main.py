from flask import Flask, request
import src.modules.file_manager as file_manager_module


app = Flask(__name__)

file_manager = file_manager_module.FileManager()

@app.route('/uploadFile', methods = ['POST'])
def uploadFile():
    return file_manager.uploadFile()

@app.route('/uploadMediaFile', methods = ['POST'])
def uploadMediaFile():
    return file_manager.uploadMediaFile()

@app.route('/webFileUpload', methods = ['POST'])
def webFileUpload():
    return file_manager.webFileUpload()

if __name__ == "__main__":
    app.run(debug=True)