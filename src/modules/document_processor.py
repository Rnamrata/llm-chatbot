import os
import re
from src import config
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from PyPDF2 import PdfReader



def clean_text(text):
    """
    Makes extracted text safe to embed and store. PDF readers sometimes split a
    character outside the basic range (maths letters like 𝜀, emoji) into two halves,
    or keep only one half. A pair is joined back into the real character; a lone half
    can't be encoded as UTF-8 (so Ollama and Chroma reject the whole file) and is dropped.
    """
    if not any('\ud800' <= c <= '\udfff' for c in text):
        return text
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'replace').replace('\ufffd', '')
# Parts of a web page that are never the article: menus, banners, forms, scripts
NOISE_TAGS = ['script', 'style', 'noscript', 'template', 'svg', 'iframe', 'nav', 'aside', 'form', 'button', 'select', 'dialog']
NOISE_ROLES = {'navigation', 'banner', 'contentinfo', 'search', 'dialog', 'complementary'}
# Words in a class or id that mark site furniture ("site-nav", "cookie_banner").
# Matched as whole words, so "article-body" or "navigator" stay.
NOISE_WORDS = {
    'nav', 'navbar', 'navigation', 'menu', 'breadcrumb', 'breadcrumbs', 'sidebar', 'footer',
    'cookie', 'cookies', 'consent', 'subscribe', 'newsletter', 'share', 'sharing', 'social',
    'related', 'recommended', 'promo', 'advert', 'ad', 'ads', 'modal', 'popup', 'banner', 'skip',
    'cta', 'archive', 'archives', 'explore', 'contact',
}


def _is_noise(tag):
    if tag.get('aria-hidden') == 'true' or (tag.get('role') or '').lower() in NOISE_ROLES:
        return True
    names = ' '.join(tag.get('class') or []) + ' ' + (tag.get('id') or '')
    return bool(NOISE_WORDS & set(re.split(r'[^a-z0-9]+', names.lower())))


def clean_web_page(soup):
    """
    The readable text of a web page, without its navigation, footers and banners
    (which otherwise end up in every passage: "All articles · Become a member").

    Args:
        soup: The page as BeautifulSoup, e.g. from WebBaseLoader.scrape()

    Returns:
        tuple: (title, text)
    """
    title = soup.title.get_text(' ', strip=True) if soup.title else ''

    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()
    # A page's own header and footer go; an article's header (its headline) stays
    for tag in soup.find_all(['header', 'footer']):
        if not tag.find_parent(['article', 'main']):
            tag.decompose()
    for tag in [t for t in soup.find_all(True) if t.attrs is not None and _is_noise(t)]:
        if not tag.decomposed:
            tag.decompose()

    # The article itself when the page marks it, else the whole body
    articles = soup.find_all('article')
    root = (
        max(articles, key=lambda a: len(a.get_text())) if articles
        else soup.find('main') or soup.find(attrs={'role': 'main'}) or soup.body or soup
    )

    lines = (' '.join(line.split()) for line in root.get_text('\n').splitlines())
    return title, '\n'.join(line for line in lines if line)


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
        
        content = clean_text(content)
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
    