from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import request
from langchain.text_splitter import MarkdownHeaderTextSplitter

class DocumentProcessor:

    def chunkingDocument(self, doc):

        return