# src/modules/llm_manager.py
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from src import config

DOCS_PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following context from documents and the chat history to answer the question.
If you cannot answer based on the context provided, say so clearly.
Each passage starts with its source in square brackets: the source's name, what kind it is and when it was added. Use it to tell which source the question means ("the PDF", "the video", "the newest one").
When you say where something comes from, use the source's name as it appears in the brackets, in plain words. Do not copy the square brackets, the kind or the date, do not name a source that is not in the passages, and do not start your answer with a source.
When a passage lists several facts or numbers, give all of them.
The current question may refer to earlier messages ("it", "that part", "only the changes"); use the chat history to understand what it means, and answer the current question rather than repeating an earlier answer.

Context from documents:
{context}

Chat History:
{chat_history}

Current Question: {question}

Helpful Answer:"""

CODE_REVIEW_PROMPT_TEMPLATE = """You are a senior engineer helping a developer with one file of their code and its automated review.
Below are the file, with its line number at the start of every line, and the findings of the review.

How to answer:
- Answer any question about this code, not only about the findings. If no finding covers what is asked, read the code and answer from it.
- The findings can miss problems or name the wrong line. When asked whether something is wrong, read those lines yourself: look for values that can be null or missing, a value used before it is checked, off-by-one ranges and loop bounds, division by zero, and errors that are ignored.
- When you name a line, use the number printed at the start of that line below. Never count lines yourself.
- Only describe problems you can see in the code below. Do not invent issues, functions, methods or library calls.
- Write any fix in the same language as the file, as code that would compile or run in that file. Do not switch to another language or syntax the file does not use.
- If the file shown is not the complete file, or the question cannot be answered from it, say so.
- The current question may refer to earlier messages ("that", "line 12", "only the changes"); use the chat history to understand it, and answer the current question rather than repeating an earlier answer. When asked for only the changes, show just the changed lines.
- A fix changes only what is broken. Keep the function's name, arguments and what they mean (for example whether a page number starts at 0 or 1) unless the developer asks to change them.
- Do not show example calls or output unless you work them out from the code; if you do, show each step ("start = 2 * 3 = 6").
- For a fix, say where each change goes using the current line numbers ("Replace line 35 with:", "Insert after line 31:"); do not number new lines yourself.
- Name errors as the file's language does (TypeError in TypeScript, NullPointerException in Java, ZeroDivisionError in Python).
- Be concise.

{context}

Chat History:
{chat_history}

Current Question: {question}

Answer:"""

# How each passage is shown to the model: its source first (set by DocumentRetriever)
DOCUMENT_PROMPT = PromptTemplate(
    template="[{source_label}]\n{page_content}",
    input_variables=["page_content", "source_label"],
)

PROMPT_TEMPLATES = {
    "docs": DOCS_PROMPT_TEMPLATE,
    "code_review": CODE_REVIEW_PROMPT_TEMPLATE,
}


def history_text(messages):
    """Chat history as the prompt shows it: 'Developer: ...' and 'Assistant: ...' lines"""
    turns = []
    for m in messages:
        who = 'Developer' if m.type == 'human' else 'Assistant'
        turns.append(f'{who}: {m.content}')
    return '\n\n'.join(turns) if turns else '(none)'


class ReviewChat:
    """
    The Ask Lens chain: called like ConversationalRetrievalChain, with
    {"question": ...}, and returns {"answer", "source_documents"}.

    Unlike that chain, it does not first ask the model to rewrite a follow-up as a
    standalone question. The retriever returns the whole review whatever the
    question, so the rewrite only cost time and changed what was asked.
    """

    def __init__(self, llm, retriever, prompt, memory, history_messages=None):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.memory = memory
        self.history_messages = history_messages or config.REVIEW_HISTORY_MESSAGES

    def __call__(self, inputs):
        question = inputs['question']
        docs = self.retriever.invoke(question)
        context = '\n\n'.join(d.page_content for d in docs)
        history = self.memory.chat_memory.messages[-self.history_messages:]

        answer = self.llm.invoke(self.prompt.format(
            context=context, chat_history=history_text(history), question=question
        ))
        self.memory.save_context({'question': question}, {'answer': answer})
        return {'answer': answer, 'source_documents': docs}


class LLMManager:
    """Manages LLM initialization and query processing"""

    def __init__(self, model_name=None, temperature=None):
        """
        Initialize LLM Manager

         Args:
            model_name: Name of the Ollama model to use (defaults to config.LLM_MODEL)
            temperature: Temperature for general chat (defaults to config.LLM_TEMPERATURE)
        """
        self.model_name = model_name or config.LLM_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.llm = OllamaLLM(model=self.model_name, temperature=self.temperature, num_ctx=config.LLM_NUM_CTX)
        self.review_model_name = config.REVIEW_LLM_MODEL
        self.review_llm = OllamaLLM(
            model=self.review_model_name,
            temperature=config.REVIEW_LLM_TEMPERATURE,
            seed=config.REVIEW_LLM_SEED,
            num_ctx=config.REVIEW_NUM_CTX,
        )

    def create_conversational_chain(self, retriever, mode="docs"):
        """
        Create a conversational chain

          Args:
            retriever: Retriever for the context. For "code_review", a
                       ReviewContextRetriever that returns the whole review.
            mode: "docs" for general document Q&A, or "code_review" for Ask Lens
                  (the numbered file and its findings)

        Returns:
            tuple: (chain, memory)
        """
        if mode not in PROMPT_TEMPLATES:
            raise ValueError(f"Unknown mode: {mode}")

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATES[mode],
            input_variables=["context", "chat_history", "question"]
        )

        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        if mode == "code_review":
            return ReviewChat(self.review_llm, retriever, prompt, memory), memory

        # Document chats still rewrite a follow-up to search with ("it" finds
        # nothing), but the answer is written to the question as it was asked
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            rephrase_question=False,
            combine_docs_chain_kwargs={"prompt": prompt, "document_prompt": DOCUMENT_PROMPT},
            verbose=False
        )

        return chain, memory

    def simple_query(self, prompt):
        """
        Simple query without retrieval (direct LLM call)

        Args:
            prompt: The prompt to send to LLM

        Returns:
            str: LLM response
        """
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error in simple query: {e}")
            return f"Error: {str(e)}"

    def format_sources(self, source_documents, max_length=200):
        """
        Format source documents for response

        Args:
            source_documents: List of retrieved documents
            max_length: Maximum length for content preview

        Returns:
            list: Formatted source information
        """
        sources = []
        for doc in source_documents:
            content = doc.page_content
            preview = content[:max_length] + "..." if len(content) > max_length else content

            source_info = {
                'content': preview,
                'metadata': doc.metadata,
                'full_length': len(content)
            }
            sources.append(source_info)

        return sources
