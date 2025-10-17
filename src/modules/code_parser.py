"""
Code Parser Module
Handles syntax-aware parsing and chunking of source code files.
Preserves code structure (functions, classes) while chunking.
"""

import re
from typing import List, Dict, Tuple
from langchain.docstore.document import Document


class CodeParser:
    """
    Parser for source code files with language-specific handling.
    Chunks code intelligently while preserving structure.
    """

    # Supported file extensions and their languages
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.cs': 'csharp',
        '.html': 'html',
        '.css': 'css',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
    }

    def __init__(self, max_chunk_size=2000):
        """
        Initialize the code parser.

        Args:
            max_chunk_size: Maximum size of each code chunk in characters
        """
        self.max_chunk_size = max_chunk_size

    def detect_language(self, filename: str) -> str:
        """
        Detect programming language from file extension.

        Args:
            filename: Name of the file

        Returns:
            Language name or 'unknown'
        """
        extension = '.' + filename.split('.')[-1] if '.' in filename else ''
        return self.LANGUAGE_EXTENSIONS.get(extension.lower(), 'unknown')

    def is_code_file(self, filename: str) -> bool:
        """
        Check if a file is a supported code file.

        Args:
            filename: Name of the file

        Returns:
            True if file is a supported code file
        """
        return self.detect_language(filename) != 'unknown'

    def extract_python_functions(self, code: str) -> List[Dict]:
        """
        Extract function and class definitions from Python code.

        Args:
            code: Python source code

        Returns:
            List of dictionaries with function/class info
        """
        structures = []
        lines = code.split('\n')
        current_indent = 0
        current_structure = None
        start_line = 0

        for i, line in enumerate(lines):
            # Detect function or class definition
            match = re.match(r'^(\s*)(def|class)\s+(\w+)', line)
            if match:
                # Save previous structure if exists
                if current_structure:
                    structures.append({
                        'type': current_structure['type'],
                        'name': current_structure['name'],
                        'start_line': start_line,
                        'end_line': i - 1,
                        'code': '\n'.join(lines[start_line:i])
                    })

                # Start new structure
                current_indent = len(match.group(1))
                current_structure = {
                    'type': match.group(2),
                    'name': match.group(3)
                }
                start_line = i
            elif current_structure and line.strip() and not line.startswith(' ' * (current_indent + 1)) and not line.strip().startswith('#'):
                # End of current structure (dedent detected)
                if line[0] not in (' ', '\t'):
                    structures.append({
                        'type': current_structure['type'],
                        'name': current_structure['name'],
                        'start_line': start_line,
                        'end_line': i - 1,
                        'code': '\n'.join(lines[start_line:i])
                    })
                    current_structure = None

        # Add last structure
        if current_structure:
            structures.append({
                'type': current_structure['type'],
                'name': current_structure['name'],
                'start_line': start_line,
                'end_line': len(lines) - 1,
                'code': '\n'.join(lines[start_line:])
            })

        return structures

    def extract_javascript_functions(self, code: str) -> List[Dict]:
        """
        Extract function and class definitions from JavaScript/TypeScript code.

        Args:
            code: JavaScript/TypeScript source code

        Returns:
            List of dictionaries with function/class info
        """
        structures = []

        # Match function declarations, arrow functions, and classes
        patterns = [
            (r'function\s+(\w+)\s*\([^)]*\)\s*{', 'function'),
            (r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', 'arrow_function'),
            (r'class\s+(\w+)\s*{', 'class'),
            (r'(\w+)\s*:\s*function\s*\([^)]*\)\s*{', 'method'),
        ]

        for pattern, struct_type in patterns:
            for match in re.finditer(pattern, code):
                structures.append({
                    'type': struct_type,
                    'name': match.group(1),
                    'start': match.start(),
                    'match': match.group(0)
                })

        return structures

    def extract_imports(self, code: str, language: str) -> List[str]:
        """
        Extract import statements from code.

        Args:
            code: Source code
            language: Programming language

        Returns:
            List of import statements
        """
        imports = []

        if language == 'python':
            imports = re.findall(r'^(?:from\s+[\w.]+\s+)?import\s+.+$', code, re.MULTILINE)
        elif language in ['javascript', 'typescript']:
            imports = re.findall(r'^import\s+.+$', code, re.MULTILINE)
        elif language == 'java':
            imports = re.findall(r'^import\s+[\w.]+;$', code, re.MULTILINE)
        elif language == 'go':
            # Go import block
            import_block = re.search(r'import\s+\(([^)]+)\)', code)
            if import_block:
                imports = [line.strip() for line in import_block.group(1).split('\n') if line.strip()]
            else:
                imports = re.findall(r'import\s+"[^"]+"', code)

        return imports

    def chunk_code(self, code: str, filename: str) -> List[Document]:
        """
        Chunk code intelligently based on language and structure.

        Args:
            code: Source code content
            filename: Name of the source file

        Returns:
            List of Document objects with code chunks
        """
        language = self.detect_language(filename)
        chunks = []

        # Extract imports
        imports = self.extract_imports(code, language)
        imports_text = '\n'.join(imports) if imports else ''

        # Language-specific chunking
        if language == 'python':
            structures = self.extract_python_functions(code)

            # If we found structures, chunk by function/class
            if structures:
                for struct in structures:
                    chunk_content = struct['code']

                    # Include imports if chunk is large enough
                    if imports_text and len(chunk_content) + len(imports_text) < self.max_chunk_size:
                        chunk_content = imports_text + '\n\n' + chunk_content

                    chunks.append(Document(
                        page_content=chunk_content,
                        metadata={
                            'source': filename,
                            'language': language,
                            'content_type': 'code',
                            'structure_type': struct['type'],
                            'structure_name': struct['name'],
                            'start_line': struct['start_line'],
                            'end_line': struct['end_line']
                        }
                    ))
            else:
                # No structures found, chunk by size
                chunks = self._chunk_by_size(code, filename, language)

        elif language in ['javascript', 'typescript']:
            structures = self.extract_javascript_functions(code)

            if structures:
                # For JS/TS, we'll chunk by size but with structure awareness
                chunks = self._chunk_by_size(code, filename, language, structures)
            else:
                chunks = self._chunk_by_size(code, filename, language)

        else:
            # Generic chunking for other languages
            chunks = self._chunk_by_size(code, filename, language)

        # If no chunks created, create at least one
        if not chunks:
            chunks.append(Document(
                page_content=code,
                metadata={
                    'source': filename,
                    'language': language,
                    'content_type': 'code'
                }
            ))

        return chunks

    def _chunk_by_size(self, code: str, filename: str, language: str,
                       structures: List[Dict] = None) -> List[Document]:
        """
        Chunk code by size with line awareness.

        Args:
            code: Source code
            filename: Filename
            language: Programming language
            structures: Optional list of code structures

        Returns:
            List of Document chunks
        """
        chunks = []
        lines = code.split('\n')
        current_chunk = []
        current_size = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(Document(
                    page_content='\n'.join(current_chunk),
                    metadata={
                        'source': filename,
                        'language': language,
                        'content_type': 'code',
                        'start_line': start_line,
                        'end_line': i - 1
                    }
                ))
                current_chunk = []
                current_size = 0
                start_line = i

            current_chunk.append(line)
            current_size += line_size

        # Add remaining chunk
        if current_chunk:
            chunks.append(Document(
                page_content='\n'.join(current_chunk),
                metadata={
                    'source': filename,
                    'language': language,
                    'content_type': 'code',
                    'start_line': start_line,
                    'end_line': len(lines) - 1
                }
            ))

        return chunks

    def calculate_complexity(self, code: str, language: str) -> Dict:
        """
        Calculate basic code complexity metrics.

        Args:
            code: Source code
            language: Programming language

        Returns:
            Dictionary with complexity metrics
        """
        metrics = {
            'lines_of_code': len(code.split('\n')),
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'cyclomatic_complexity': 0
        }

        if language == 'python':
            metrics['num_functions'] = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
            metrics['num_classes'] = len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE))
            metrics['num_imports'] = len(re.findall(r'^(?:from\s+[\w.]+\s+)?import\s+', code, re.MULTILINE))
            # Simple cyclomatic complexity (count decision points)
            metrics['cyclomatic_complexity'] = len(re.findall(r'\b(if|elif|for|while|except|and|or)\b', code))

        elif language in ['javascript', 'typescript']:
            metrics['num_functions'] = len(re.findall(r'function\s+\w+|=>\s*{', code))
            metrics['num_classes'] = len(re.findall(r'class\s+\w+', code))
            metrics['num_imports'] = len(re.findall(r'^import\s+', code, re.MULTILINE))
            metrics['cyclomatic_complexity'] = len(re.findall(r'\b(if|else if|for|while|case|&&|\|\|)\b', code))

        return metrics
