import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents_from_folder(folder_path: str):
    documents = []

    # Recursively walk through all subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            if filename.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

            elif filename.endswith(".txt") or filename.endswith(".md"):
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return documents


if __name__ == "__main__":
    folder_path = "data/docs"
    docs = load_documents_from_folder(folder_path)

    print(f"Loaded {len(docs)} documents\n")

    # Print sample content
    for i, doc in enumerate(docs[:5]):
        print(f"\n--- Document {i+1} ---")
        print("Length:", len(doc.page_content))
        print("Content preview:", repr(doc.page_content[:200]))