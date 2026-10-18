from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DB_DIR = PROJECT_DIR / "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in DATA_DIR.glob("*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path.name).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    store = None

    if DB_DIR.exists():
        store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        if store.get()["ids"]:
            return store

    docs = load_transcripts()
    if not docs:
        raise FileNotFoundError(f"No .vtt transcripts found in {DATA_DIR}")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    if store is not None:
        store.add_documents(chunks)
        return store

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":
    print("Building retriever...", flush=True)
    retriever = build_retriever()

    print("Searching...", flush=True)
    results = retriever.invoke("what is regression testing?")

    print(f"Found {len(results)} results.", flush=True)
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")