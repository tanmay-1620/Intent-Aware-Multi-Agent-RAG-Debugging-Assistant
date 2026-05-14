import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.load_docs import load_documents_from_folder
from ingestion.load_code import load_code_from_folder
from embeddings.embedding_model import load_embedding_model


# 🔥 Add structured metadata (VERY IMPORTANT)
def enrich_metadata(documents, source_type):
    enriched_docs = []

    for doc in documents:
        metadata = doc.metadata or {}

        metadata["type"] = source_type

        if source_type == "code":
            file_path = metadata.get("source", "")
            metadata["filename"] = os.path.basename(file_path)

        enriched_docs.append(
            type(doc)(
                page_content=doc.page_content,
                metadata=metadata
            )
        )

    return enriched_docs


def load_all_data(
    docs_path="data/docs",
    code_path="data/code",
    extra_path="debug_dataset"   
):
    docs = load_documents_from_folder(docs_path)
    code_docs = load_code_from_folder(code_path)
    extra_docs = load_documents_from_folder(extra_path)

    docs = enrich_metadata(docs, "documentation")
    code_docs = enrich_metadata(code_docs, "code")
    extra_docs = enrich_metadata(extra_docs, "debug")

    print(f"📄 Loaded {len(docs)} documentation files")
    print(f"💻 Loaded {len(code_docs)} code files")
    print(f"🔧 Loaded {len(extra_docs)} debug files")

    all_docs = docs + code_docs + extra_docs
    print(f"📊 Total documents: {len(all_docs)}")

    return all_docs


# 🔥 Smart chunking (DIFFERENT for code vs docs)
def split_documents(documents):
    split_docs = []

    for doc in documents:
        if doc.metadata.get("type") == "code":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=80,
                separators=["\nclass ", "\ndef ", "\n\n", "\n"]
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=120,
                separators=["\n\n", "\n", " ", ""]
            )

        chunks = splitter.split_documents([doc])
        split_docs.extend(chunks)

    print(f"✂️ Split into {len(split_docs)} chunks")

    return split_docs


def create_vector_store(docs_path="data/docs", code_path="data/code", extra_path="debug_dataset"):
    print("\n🚀 Creating Vector Store...\n")

    # Step 1: Load data
    documents = load_all_data(docs_path, code_path, extra_path)

    # Step 2: Split
    split_docs = split_documents(documents)

    # Step 3: Embeddings
    print("\n🔎 Loading embedding model...")
    embeddings = load_embedding_model()

    # Step 4: Create vector DB
    print("\n📦 Creating FAISS index...")
    vector_store = FAISS.from_documents(split_docs, embeddings)

    print("\n✅ Vector store created successfully!")

    return vector_store


def save_vector_store(vector_store, path="vectorstore/faiss_index"):
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    print(f"💾 Vector store saved at: {path}")


def load_vector_store(path="vectorstore/faiss_index"):
    print("\n📂 Loading vector store...")
    embeddings = load_embedding_model()

    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✅ Vector store loaded successfully!")

    return vector_store


def test_retrieval(vector_store):
    print("\n🧪 Testing retrieval...\n")

    test_queries = [
        "Where is login API defined?",
        "Explain request flow",
        "What is FastAPI?"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")

        results = vector_store.similarity_search(query, k=3)

        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Type: {doc.metadata.get('type')}")
            print(f"File: {doc.metadata.get('filename')}")
            print(doc.page_content[:200])


if __name__ == "__main__":
    vs = create_vector_store()
    save_vector_store(vs)
    test_retrieval(vs)