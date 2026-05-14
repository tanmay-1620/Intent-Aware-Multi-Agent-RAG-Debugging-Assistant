from langchain_community.embeddings import HuggingFaceEmbeddings


def load_embedding_model():
    """
    Loads a lightweight and fast embedding model
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    return embeddings


if __name__ == "__main__":
    # Test embedding model
    model = load_embedding_model()

    sample_text = "FastAPI is a modern Python web framework."

    vector = model.embed_query(sample_text)

    print("Sample text:", sample_text)
    print("Vector length:", len(vector))
    print("First 5 values:", vector[:5])