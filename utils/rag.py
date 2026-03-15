import os
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _load_single_file(file_path: str) -> List[Document]:
    try:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            return PyPDFLoader(file_path).load()
        if extension in {".txt", ".md"}:
            return TextLoader(file_path, encoding="utf-8").load()
        return []
    except Exception as error:
        return [Document(page_content=f"Failed to read {file_path}: {error}", metadata={"source": file_path, "error": True})]


def load_documents_from_directory(directory_path: str) -> Tuple[List[Document], List[str]]:
    try:
        if not os.path.isdir(directory_path):
            return [], [f"Knowledge base directory not found: {directory_path}"]

        documents: List[Document] = []
        warnings: List[str] = []

        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                extension = os.path.splitext(file_path)[1].lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    continue

                loaded_docs = _load_single_file(file_path)
                if not loaded_docs:
                    warnings.append(f"Skipped file: {file_path}")
                    continue

                documents.extend(loaded_docs)

        return documents, warnings
    except Exception as error:
        return [], [f"Failed to load knowledge base: {error}"]


def build_retriever(documents: List[Document], embedding_model, chunk_size: int = 800, chunk_overlap: int = 120, top_k: int = 4):
    try:
        if not documents:
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            return None

        vector_store = FAISS.from_documents(chunks, embedding_model)
        return vector_store.as_retriever(search_kwargs={"k": top_k})
    except Exception:
        return None


def retrieve_context(retriever, query: str) -> str:
    try:
        if retriever is None:
            return ""

        docs = retriever.invoke(query)
        if not docs:
            return ""

        contexts = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            contexts.append(f"[{index}] Source: {source}\n{doc.page_content}")

        return "\n\n".join(contexts)
    except Exception:
        return ""
