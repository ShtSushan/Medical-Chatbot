from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from helper import load_pdf_files, filter_to_minimal_doc, text_splitter, download_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load and process documents
extracted_data = load_pdf_files("../Data")
minimal_docs = filter_to_minimal_doc(extracted_data)
texts_chunk = text_splitter(minimal_docs)
embeddings = download_embeddings()

# Setup Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Upload documents ⚠️ Run this file ONCE only!
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)

print("Documents uploaded successfully!")