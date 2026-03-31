from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf_files(Data):
    loader = DirectoryLoader(
        Data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_doc(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


def text_splitter(minimal_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    texts_chunk = splitter.split_documents(minimal_docs)
    return texts_chunk


def download_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings