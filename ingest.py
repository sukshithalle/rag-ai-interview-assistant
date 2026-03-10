import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH = "data"


documents = []

# Load all files in data folder
for file in os.listdir(DATA_PATH):

    path = os.path.join(DATA_PATH, file)

    if file.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())

    elif file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())


# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create vector database
db = FAISS.from_documents(chunks, embeddings)

db.save_local("vector_db")


print("All documents indexed successfully")