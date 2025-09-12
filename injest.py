import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")



def load_documents():
    """Load all documents from the data directory."""
    docs = []
    import glob

    for file in glob.glob("data/*.pdf"):
        print(f"Processing pdf file: {file}")
        Loader = PyPDFLoader(file)
        doc = Loader.load()
        for d in doc:
            d.metadata.update({"source": file})
        docs.extend(doc)

    for file in glob.glob("data/*.txt"):
        print(f"Processing text file: {file}")
        Loader = TextLoader(file, encoding='utf8')
        doc = Loader.load()
        for d in doc:
            d.metadata.update({"source": file})
        docs.extend(doc)
    
    for file in glob.glob("data/*.csv"):
        print(f"Processing csv file: {file}")
        Loader = CSVLoader(file, encoding="utf-8")
        doc = Loader.load()
        for d in doc:
            d.metadata.update({"source": file})
        docs.extend(doc)

    return docs

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def store_embeddings(chunks):
    """Create and store embeddings in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR 
    )
    vectordb
    print(f"Stored {len(chunks)} chunks in Chroma at {PERSIST_DIR}") 
    return vectordb

if __name__ =="__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} raw documents")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    
    store_embeddings(chunks)








