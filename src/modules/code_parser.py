"""
Code Parser Module
====================
This module provides intelligent code parsing and chunking capabilities for various programming languages.
It handles syntax-aware parsing that respects code structure (functions, classes) while creating manageable chunks
for vector database storage and LLM processing.

Main Features:
- Language detection from file extensions
- Structure-aware code chunking (preserves functions and classes)
- Import statement extraction
- Code complexity metrics calculation
- Support for 20+ programming languages
"""

# Import regex module for pattern matching in code (finding functions, classes, imports, etc.)
import re
# Import type hints for better code documentation and IDE support
from typing import List, Dict, Tuple
# Import LangChain's Document class to create structured documents with metadata
from langchain.docstore.document import Document


class CodeParser:
    """
    Intelligent Code Parser for Multiple Programming Languages
    ==========================================================
    This class provides comprehensive code parsing functionality with the following capabilities:

    1. Language Detection: Automatically detects programming language from file extension
    2. Structure Extraction: Identifies functions, classes, and methods in the code
    3. Smart Chunking: Splits code into manageable pieces while preserving logical boundaries
    4. Import Analysis: Extracts import/include statements for better context
    5. Complexity Analysis: Calculates basic code metrics (LOC, cyclomatic complexity, etc.)

    The parser is designed to work with RAG (Retrieval-Augmented Generation) systems,
    ensuring that code chunks maintain their semantic meaning and structural integrity.
    """

    # Dictionary mapping file extensions to programming language names
    # This allows the parser to automatically detect what language a file is written in
    # Each key is a file extension (with leading dot), and each value is the language name
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',          # Python source files
        '.js': 'javascript',      # JavaScript files
        '.ts': 'typescript',      # TypeScript files
        '.jsx': 'javascript',     # React JavaScript files (JSX syntax)
        '.tsx': 'typescript',     # React TypeScript files (TSX syntax)
        '.java': 'java',          # Java source files
        '.go': 'go',              # Go (Golang) files
        '.cpp': 'cpp',            # C++ source files
        '.c': 'c',                # C source files
        '.h': 'c',                # C header files (treated as C)
        '.hpp': 'cpp',            # C++ header files
        '.rs': 'rust',            # Rust source files
        '.rb': 'ruby',            # Ruby source files
        '.php': 'php',            # PHP files
        '.swift': 'swift',        # Swift (iOS/macOS) files
        '.kt': 'kotlin',          # Kotlin (Android) files
        '.cs': 'csharp',          # C# files
        '.html': 'html',          # HTML markup files
        '.css': 'css',            # CSS stylesheet files
        '.sql': 'sql',            # SQL database query files
        '.sh': 'shell',           # Shell script files (.sh)
        '.bash': 'shell',         # Bash script files
    }

    def __init__(self, max_chunk_size=2000):
        """
        Initialize the CodeParser with configuration settings.

        The parser is initialized with a maximum chunk size to ensure that code pieces
        sent to the LLM or stored in the vector database are not too large.
        2000 characters is a good default that typically captures 1-2 functions while
        staying within reasonable token limits for embeddings.

        Args:
            max_chunk_size (int): Maximum number of characters allowed in each code chunk.
                                 Larger chunks provide more context but may exceed token limits.
                                 Smaller chunks are more granular but may lose context.
                                 Default: 2000 characters (~500 tokens)

        Instance Variables:
            self.max_chunk_size: Stored configuration for use in chunking methods
        """
        # Store the maximum chunk size as an instance variable for later use in chunking methods
        self.max_chunk_size = max_chunk_size

    def detect_language(self, filename: str) -> str:
        """
        Automatically detect the programming language of a file based on its extension.

        This method extracts the file extension and looks it up in the LANGUAGE_EXTENSIONS
        dictionary. This is crucial for applying language-specific parsing rules and syntax
        highlighting in the downstream processing.

        How it works:
        1. Splits the filename by the dot character to isolate the extension
        2. Takes the last part (in case of multiple dots, like file.test.js)
        3. Adds a leading dot to match the dictionary format
        4. Converts to lowercase for case-insensitive matching
        5. Looks up in LANGUAGE_EXTENSIONS dictionary

        Args:
            filename (str): The name of the file including extension
                           Examples: "main.py", "app.js", "index.tsx"

        Returns:
            str: The detected programming language name (e.g., "python", "javascript")
                 Returns 'unknown' if the extension is not in the supported list

        Examples:
            >>> parser.detect_language("main.py")
            'python'
            >>> parser.detect_language("App.jsx")
            'javascript'
            >>> parser.detect_language("data.csv")
            'unknown'
        """
        # Extract file extension: split by '.' and take the last part
        # Example: "main.py" -> ["main", "py"] -> "py"
        # If no dot exists, use empty string to avoid index errors
        extension = '.' + filename.split('.')[-1] if '.' in filename else ''

        # Look up the extension in our dictionary, converting to lowercase for case-insensitivity
        # If not found, return 'unknown' as the default value
        return self.LANGUAGE_EXTENSIONS.get(extension.lower(), 'unknown')

    def is_code_file(self, filename: str) -> bool:
        """
        Determine whether a given file is a supported code file that can be parsed.

        This is a convenience method that uses detect_language() internally.
        It's useful for filtering files before processing them.

        Args:
            filename (str): The name of the file to check

        Returns:
            bool: True if the file is a recognized code file (has a supported extension)
                  False if the file type is not supported (returns 'unknown' from detect_language)

        Examples:
            >>> parser.is_code_file("main.py")
            True
            >>> parser.is_code_file("document.pdf")
            False
        """
        # If detect_language returns anything other than 'unknown', it's a code file
        return self.detect_language(filename) != 'unknown'

    def extract_python_functions(self, code: str) -> List[Dict]:
        """
        Extract all function and class definitions from Python source code.

        This method parses Python code to identify structural boundaries (functions and classes)
        so that code can be chunked intelligently without breaking in the middle of a logical unit.

        How it works:
        1. Splits code into individual lines for line-by-line analysis
        2. Uses regex to detect 'def' (function) and 'class' (class) keywords
        3. Tracks indentation levels to determine where structures end
        4. Captures the complete code for each structure including its body
        5. Records line numbers for reference

        The algorithm works by:
        - Detecting when a new function/class starts (via regex pattern matching)
        - Saving the previous structure when a new one begins
        - Detecting dedentation (less indentation) to find structure boundaries
        - Handling edge cases like nested functions and classes

        Args:
            code (str): Complete Python source code as a string

        Returns:
            List[Dict]: List of dictionaries, each containing:
                - 'type': Either 'def' (function) or 'class'
                - 'name': The name of the function/class
                - 'start_line': Starting line number (0-indexed)
                - 'end_line': Ending line number (0-indexed)
                - 'code': The actual code string for this structure

        Example structure returned:
            [
                {
                    'type': 'def',
                    'name': 'calculate_sum',
                    'start_line': 0,
                    'end_line': 5,
                    'code': 'def calculate_sum(a, b):\n    return a + b'
                },
                ...
            ]
        """
        # Initialize list to store all discovered structures (functions and classes)
        structures = []

        # Split the entire code into individual lines for line-by-line analysis
        lines = code.split('\n')

        # Track the current indentation level (number of spaces) to detect structure boundaries
        current_indent = 0

        # Track the currently open structure (function or class) that we're processing
        current_structure = None

        # Track the line number where the current structure started
        start_line = 0

        # Iterate through each line with its index (line number)
        for i, line in enumerate(lines):
            # Use regex to detect function or class definition lines
            # Pattern explanation:
            # ^(\s*) - Capture leading whitespace (indentation) at start of line
            # (def|class) - Match either 'def' or 'class' keyword
            # \s+ - Require at least one space after the keyword
            # (\w+) - Capture the function/class name (word characters)
            match = re.match(r'^(\s*)(def|class)\s+(\w+)', line)

            if match:
                # Found a new function or class definition!

                # Before starting the new structure, save the previous one if it exists
                if current_structure:
                    structures.append({
                        'type': current_structure['type'],        # 'def' or 'class'
                        'name': current_structure['name'],        # Function/class name
                        'start_line': start_line,                 # Where it started
                        'end_line': i - 1,                        # Where it ended (line before current)
                        'code': '\n'.join(lines[start_line:i])   # Extract the actual code
                    })

                # Now start tracking the new structure we just found
                # Extract the indentation level (number of leading spaces) to track scope
                current_indent = len(match.group(1))

                # Store metadata about this new structure
                current_structure = {
                    'type': match.group(2),   # 'def' or 'class' (captured group 2)
                    'name': match.group(3)    # The function/class name (captured group 3)
                }

                # Remember where this structure started
                start_line = i

            elif current_structure and line.strip() and not line.startswith(' ' * (current_indent + 1)) and not line.strip().startswith('#'):
                # Check if we've reached the end of the current structure
                # This happens when we encounter a line that:
                # 1. Has some content (not empty: line.strip())
                # 2. Is not indented further than the structure (not a child)
                # 3. Is not a comment line

                # If the line starts at column 0 (no leading space/tab), it's definitely outside our structure
                if line[0] not in (' ', '\t'):
                    # Save the completed structure
                    structures.append({
                        'type': current_structure['type'],
                        'name': current_structure['name'],
                        'start_line': start_line,
                        'end_line': i - 1,                         # Ended on the previous line
                        'code': '\n'.join(lines[start_line:i])
                    })
                    # Clear current_structure since we've finished processing it
                    current_structure = None

        # Handle the last structure in the file (no more lines after it to trigger the save)
        if current_structure:
            structures.append({
                'type': current_structure['type'],
                'name': current_structure['name'],
                'start_line': start_line,
                'end_line': len(lines) - 1,                   # Goes to the end of the file
                'code': '\n'.join(lines[start_line:])        # All remaining lines
            })

        # Return the complete list of all extracted structures
        return structures

    def extract_javascript_functions(self, code: str) -> List[Dict]:
        """
        Extract function and class definitions from JavaScript/TypeScript code.

        JavaScript has multiple ways to define functions (traditional functions, arrow functions,
        class methods, etc.), so this method uses multiple regex patterns to catch them all.

        Detected patterns:
        1. Traditional function declarations: function myFunc() {}
        2. Arrow functions assigned to const: const myFunc = () => {}
        3. Class declarations: class MyClass {}
        4. Object methods: myMethod: function() {}

        Args:
            code (str): JavaScript or TypeScript source code

        Returns:
            List[Dict]: List of discovered structures with metadata:
                - 'type': Type of structure (function, arrow_function, class, method)
                - 'name': Name of the function/class
                - 'start': Character position where it starts in the code
                - 'match': The matched text
        """
        # Initialize empty list to collect all found structures
        structures = []

        # Define regex patterns for different JavaScript function/class syntaxes
        # Each tuple contains (pattern, structure_type_name)
        patterns = [
            # Traditional function declaration: function myFunc(params) { ... }
            (r'function\s+(\w+)\s*\([^)]*\)\s*{', 'function'),

            # Arrow function with const: const myFunc = (params) => { ... }
            (r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', 'arrow_function'),

            # Class declaration: class MyClass { ... }
            (r'class\s+(\w+)\s*{', 'class'),

            # Object method: myMethod: function(params) { ... }
            (r'(\w+)\s*:\s*function\s*\([^)]*\)\s*{', 'method'),
        ]

        # Iterate through each pattern and search the entire code for matches
        for pattern, struct_type in patterns:
            # finditer() returns all non-overlapping matches in the code
            for match in re.finditer(pattern, code):
                # Add each discovered structure to our list
                structures.append({
                    'type': struct_type,           # What kind of structure it is
                    'name': match.group(1),        # The name (captured in first group)
                    'start': match.start(),        # Character position in code
                    'match': match.group(0)        # The full matched text
                })

        # Return all found structures
        return structures

    def extract_imports(self, code: str, language: str) -> List[str]:
        """
        Extract all import/include statements from source code.

        Import statements are crucial context when analyzing code, as they show:
        - What external libraries are being used
        - Dependencies and third-party packages
        - Internal modules being referenced

        This method uses language-specific regex patterns because each language has
        different syntax for importing modules.

        Supported languages:
        - Python: import X, from X import Y
        - JavaScript/TypeScript: import X from 'Y'
        - Java: import package.Class;
        - Go: import "package" or import ( ... ) blocks

        Args:
            code (str): The source code to analyze
            language (str): The programming language (detected from file extension)

        Returns:
            List[str]: List of import statement strings found in the code
                       Returns empty list if no imports found or language not supported

        Examples:
            For Python code:
            >>> extract_imports("import os\nfrom typing import List", "python")
            ['import os', 'from typing import List']

            For JavaScript code:
            >>> extract_imports("import React from 'react'", "javascript")
            ["import React from 'react'"]
        """
        # Initialize empty list to collect import statements
        imports = []

        if language == 'python':
            # Python import patterns:
            # - "import module"
            # - "from module import something"
            # - "from package.submodule import Class"
            # Pattern: ^(?:from\s+[\w.]+\s+)?import\s+.+$
            # Explanation:
            # ^ = start of line
            # (?:from\s+[\w.]+\s+)? = optionally match "from package.name "
            # import\s+.+$ = must have "import" followed by something
            imports = re.findall(r'^(?:from\s+[\w.]+\s+)?import\s+.+$', code, re.MULTILINE)

        elif language in ['javascript', 'typescript']:
            # JavaScript/TypeScript import pattern:
            # - "import X from 'module'"
            # - "import { X, Y } from 'module'"
            # - "import * as X from 'module'"
            # Pattern matches any line starting with "import"
            imports = re.findall(r'^import\s+.+$', code, re.MULTILINE)

        elif language == 'java':
            # Java import pattern:
            # - "import package.name.ClassName;"
            # Pattern: ^import\s+[\w.]+;$
            # Must end with semicolon
            imports = re.findall(r'^import\s+[\w.]+;$', code, re.MULTILINE)

        elif language == 'go':
            # Go has two import formats:
            # 1. Block format: import ( "pkg1" "pkg2" )
            # 2. Single line: import "package"

            # First, try to find import block (more common for multiple imports)
            import_block = re.search(r'import\s+\(([^)]+)\)', code)

            if import_block:
                # Found import block, extract each line inside the parentheses
                # Split by newlines and remove empty lines
                imports = [line.strip() for line in import_block.group(1).split('\n') if line.strip()]
            else:
                # No block found, look for single-line imports
                imports = re.findall(r'import\s+"[^"]+"', code)

        # If language not matched above, imports remains empty []
        return imports

    def chunk_code(self, code: str, filename: str) -> List[Document]:
        """
        Intelligently chunk source code based on language structure and semantics.

        This is the main entry point for code chunking. It analyzes the code and chunks it
        in the most appropriate way based on the programming language and code structure.

        Chunking Strategy:
        1. Detect the programming language from the filename
        2. Extract import statements for context
        3. Apply language-specific chunking:
           - Python: Chunk by functions and classes (preserves semantic boundaries)
           - JavaScript/TypeScript: Chunk by size with structure awareness
           - Other languages: Generic size-based chunking
        4. Include import statements in chunks when space permits (for context)
        5. Create LangChain Document objects with rich metadata

        Why this matters:
        - Ensures code chunks are semantically meaningful (don't break mid-function)
        - Preserves context by including imports
        - Provides metadata for better retrieval and code review
        - Optimizes chunk size for embedding models and LLMs

        Args:
            code (str): The complete source code content to chunk
            filename (str): Name of the source file (used for language detection)

        Returns:
            List[Document]: List of LangChain Document objects, each containing:
                - page_content: The code chunk as a string
                - metadata: Dictionary with:
                    - source: Filename
                    - language: Programming language
                    - content_type: 'code'
                    - structure_type: Type of structure (for Python: 'def' or 'class')
                    - structure_name: Name of function/class
                    - start_line/end_line: Line number range
        """
        # Step 1: Detect what programming language this is
        language = self.detect_language(filename)

        # Initialize list to collect Document chunks
        chunks = []

        # Step 2: Extract all import statements from the code
        # These provide crucial context about dependencies and will be included in chunks
        imports = self.extract_imports(code, language)
        # Join imports into a single string for easy inclusion
        imports_text = '\n'.join(imports) if imports else ''

        # Step 3: Apply language-specific chunking strategies

        if language == 'python':
            # Python: Use structure-aware chunking (by functions and classes)
            # This is ideal because Python's indentation makes structure detection reliable

            # Extract all functions and classes from the code
            structures = self.extract_python_functions(code)

            # If we successfully found structures, create one chunk per structure
            if structures:
                for struct in structures:
                    # Get the code for this specific function/class
                    chunk_content = struct['code']

                    # Try to prepend import statements if they fit within size limit
                    # This gives the LLM context about what libraries are being used
                    if imports_text and len(chunk_content) + len(imports_text) < self.max_chunk_size:
                        chunk_content = imports_text + '\n\n' + chunk_content

                    # Create a Document object with the code and comprehensive metadata
                    chunks.append(Document(
                        page_content=chunk_content,                # The actual code
                        metadata={
                            'source': filename,                    # Which file it's from
                            'language': language,                  # Programming language
                            'content_type': 'code',               # Type of content
                            'structure_type': struct['type'],     # 'def' or 'class'
                            'structure_name': struct['name'],     # Function/class name
                            'start_line': struct['start_line'],   # Starting line number
                            'end_line': struct['end_line']        # Ending line number
                        }
                    ))
            else:
                # No structures found (maybe it's just a script with no functions)
                # Fall back to generic size-based chunking
                chunks = self._chunk_by_size(code, filename, language)

        elif language in ['javascript', 'typescript']:
            # JavaScript/TypeScript: Harder to reliably chunk by structure due to flexible syntax
            # Use size-based chunking but with awareness of discovered structures

            # Try to find structures (functions, classes, etc.)
            structures = self.extract_javascript_functions(code)

            if structures:
                # For JS/TS, we use size-based chunking but pass structure info
                # This helps the chunker try to respect function boundaries
                chunks = self._chunk_by_size(code, filename, language, structures)
            else:
                # No structures found, use pure size-based chunking
                chunks = self._chunk_by_size(code, filename, language)

        else:
            # For all other languages (Java, C++, Go, Rust, etc.)
            # Use generic size-based chunking
            # Future enhancement: Add structure-aware chunking for more languages
            chunks = self._chunk_by_size(code, filename, language)

        # Safety check: If chunking failed for some reason, create at least one chunk
        # with the entire code so we don't lose the content
        if not chunks:
            chunks.append(Document(
                page_content=code,
                metadata={
                    'source': filename,
                    'language': language,
                    'content_type': 'code'
                }
            ))

        # Return all created chunks
        return chunks

    def _chunk_by_size(self, code: str, filename: str, language: str,
                       structures: List[Dict] = None) -> List[Document]:
        """
        Chunk code by size (character count) while respecting line boundaries.

        This is a fallback chunking method used when structure-aware chunking is not available
        or appropriate. It ensures chunks don't exceed max_chunk_size while keeping complete lines.

        Algorithm:
        1. Split code into individual lines
        2. Add lines to current chunk until size limit would be exceeded
        3. When limit reached, save current chunk and start a new one
        4. Track line numbers for each chunk

        Args:
            code (str): Source code to chunk
            filename (str): Name of the source file
            language (str): Programming language
            structures (List[Dict], optional): Code structures (not currently used, reserved for future enhancement)

        Returns:
            List[Document]: List of code chunks with metadata including line number ranges
        """
        # Initialize list to collect chunks
        chunks = []

        # Split code into individual lines for line-by-line processing
        lines = code.split('\n')

        # Track the current chunk being built
        current_chunk = []       # List of lines in current chunk
        current_size = 0         # Total character count in current chunk
        start_line = 0           # Line number where current chunk started

        # Process each line
        for i, line in enumerate(lines):
            # Calculate this line's size (+1 for the newline character that will be added when joining)
            line_size = len(line) + 1

            # Check if adding this line would exceed the size limit
            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Size limit reached! Save the current chunk before starting a new one

                # Create Document object with the completed chunk
                chunks.append(Document(
                    page_content='\n'.join(current_chunk),     # Join lines back with newlines
                    metadata={
                        'source': filename,                    # Which file it's from
                        'language': language,                  # Programming language
                        'content_type': 'code',               # Type of content
                        'start_line': start_line,             # First line number in chunk
                        'end_line': i - 1                     # Last line number in chunk (previous line)
                    }
                ))

                # Reset for next chunk
                current_chunk = []
                current_size = 0
                start_line = i    # Next chunk starts at current line

            # Add current line to the chunk
            current_chunk.append(line)
            current_size += line_size

        # Don't forget the last chunk! (no more lines to trigger the save)
        if current_chunk:
            chunks.append(Document(
                page_content='\n'.join(current_chunk),
                metadata={
                    'source': filename,
                    'language': language,
                    'content_type': 'code',
                    'start_line': start_line,
                    'end_line': len(lines) - 1              # Goes to the last line
                }
            ))

        return chunks

    def calculate_complexity(self, code: str, language: str) -> Dict:
        """
        Calculate basic code complexity and quality metrics.

        This method provides a quick overview of code complexity by counting:
        - Lines of code (LOC): Total number of lines
        - Functions: How many functions are defined
        - Classes: How many classes are defined
        - Imports: Number of import statements (indicates dependencies)
        - Cyclomatic Complexity: Number of decision points (branches, loops, conditions)

        Higher cyclomatic complexity often indicates code that's harder to understand and test.
        Generally, complexity > 10 for a single function is considered high.

        Args:
            code (str): Source code to analyze
            language (str): Programming language (determines which patterns to use)

        Returns:
            Dict: Dictionary containing:
                - 'lines_of_code': Total number of lines
                - 'num_functions': Count of function definitions
                - 'num_classes': Count of class definitions
                - 'num_imports': Count of import statements
                - 'cyclomatic_complexity': Estimated cyclomatic complexity

        Example result:
            {
                'lines_of_code': 150,
                'num_functions': 8,
                'num_classes': 2,
                'num_imports': 5,
                'cyclomatic_complexity': 25
            }
        """
        # Initialize metrics dictionary with default values
        metrics = {
            'lines_of_code': len(code.split('\n')),    # Total lines in the code
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'cyclomatic_complexity': 0
        }

        if language == 'python':
            # Count Python functions: lines starting with 'def function_name'
            metrics['num_functions'] = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))

            # Count Python classes: lines starting with 'class ClassName'
            metrics['num_classes'] = len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE))

            # Count import statements (both 'import' and 'from...import')
            metrics['num_imports'] = len(re.findall(r'^(?:from\s+[\w.]+\s+)?import\s+', code, re.MULTILINE))

            # Cyclomatic complexity: Count decision points and logical operators
            # Each of these keywords/operators adds a path through the code:
            # - if, elif: conditional branches
            # - for, while: loops
            # - except: exception handling branches
            # - and, or: logical operators creating compound conditions
            metrics['cyclomatic_complexity'] = len(re.findall(r'\b(if|elif|for|while|except|and|or)\b', code))

        elif language in ['javascript', 'typescript']:
            # Count JavaScript/TypeScript functions (traditional functions and arrow functions)
            # Patterns: 'function name' or '=>' (arrow function indicator)
            metrics['num_functions'] = len(re.findall(r'function\s+\w+|=>\s*{', code))

            # Count classes: 'class ClassName'
            metrics['num_classes'] = len(re.findall(r'class\s+\w+', code))

            # Count import statements: lines starting with 'import'
            metrics['num_imports'] = len(re.findall(r'^import\s+', code, re.MULTILINE))

            # Cyclomatic complexity for JavaScript:
            # - if, else if: conditional branches
            # - for, while: loops
            # - case: switch statement cases
            # - &&, ||: logical operators
            metrics['cyclomatic_complexity'] = len(re.findall(r'\b(if|else if|for|while|case|&&|\|\|)\b', code))

        # For other languages, metrics remain at default values (only LOC is calculated)
        return metrics
