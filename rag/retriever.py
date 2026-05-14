from langchain_community.llms import Ollama
from vectorstore.vector_db import load_vector_store


# 🔥 AI + RULE-BASED Query Classification (IMPROVED)
def classify_query(query, llm):

    q = query.lower()

    # 🔥 HARD OVERRIDE: DEBUG detection (highest priority)
    if any(w in q for w in [
        "error", "fail", "bug", "issue", "500", "401",
        "exception", "traceback", "crash", "not working"
    ]):
        return "debug"

    prompt = f"""
Classify the user query into ONE of these categories:

- code → asking about implementation, APIs, files
- flow → asking about system process or working
- definition → asking what something is
- debug → asking about issues, errors, fixes
- general → anything else

Query:
{query}

Only return ONE word from:
code, flow, definition, debug, general
"""

    response = llm.invoke(prompt).strip().lower()

    if response not in ["code", "flow", "definition", "debug", "general"]:
        return "general"

    return response


# 🔥 Create RAG System
def create_rag_system():
    vector_store = load_vector_store()

    doc_retriever = vector_store.as_retriever(
        search_kwargs={"k": 2, "filter": {"type": "documentation"}}
    )

    code_retriever = vector_store.as_retriever(
        search_kwargs={"k": 2, "filter": {"type": "code"}}
    )

    # DEBUG retriever - selective, not greedy
    debug_retriever = vector_store.as_retriever(
        search_kwargs={"k": 2, "filter": {"type": "debug"}}
    )

    llm = Ollama(model="phi3")

    return doc_retriever, code_retriever, debug_retriever, llm


# 🔥 Intelligent Retrieval Strategy (UPDATED LOGIC)
def retrieve_documents(query, query_type, doc_retriever, code_retriever, debug_retriever):

    # 🔥 PRIORITY 1: DEBUG MODE (STRONG DOMINANCE)
    if query_type == "debug":
        debug_docs = debug_retriever.invoke(query)

        # If debug dataset is strong enough → use it
        if len(debug_docs) >= 2:
            return debug_docs

        # fallback → code retriever
        return code_retriever.invoke(query)

    # 🔥 CODE MODE
    elif query_type == "code":
        return code_retriever.invoke(query)

    # 🔥 FLOW MODE (combine code + docs)
    elif query_type == "flow":
        return code_retriever.invoke(query) + doc_retriever.invoke(query)

    # 🔥 DEFINITION MODE
    elif query_type == "definition":
        return doc_retriever.invoke(query)

    # 🔥 GENERAL FALLBACK
    return doc_retriever.invoke(query)


# 🔥 Filter weak chunks (noise reduction)
def filter_docs(docs):
    return [doc for doc in docs if len(doc.page_content.strip()) > 80]


# 🔥 Prompt Builder (dynamic instruction injection)
def build_prompt(context, query, query_type):

    if query_type == "flow":
        instruction = "Explain step-by-step using numbered points."
    elif query_type == "code":
        instruction = "Explain relevant code and mention file if possible."
    elif query_type == "debug":
        instruction = "Identify the issue clearly and suggest fixes step-by-step."
    else:
        instruction = "Answer clearly and concisely."

    return f"""
You are an AI assistant specialized in code and debugging.

STRICT RULES:
- Answer ONLY using the provided context
- DO NOT guess
- If answer is not present, say:
  "I don't have enough information to answer that."

Instructions:
{instruction}

Context:
{context}

Question:
{query}

Answer:
"""


# 🔥 Chat Loop (Main System)
def run_chat():

    doc_retriever, code_retriever, debug_retriever, llm = create_rag_system()

    print("\n🚀 Intelligent AI Debugging Assistant Ready!")
    print("Type 'exit' to quit\n")

    while True:
        query = input("💬 Ask your question: ")

        if query.lower() == "exit":
            break

        # 🔥 Step 1: Classification
        query_type = classify_query(query, llm)
        print(f"\n🧠 Detected Query Type: {query_type}")

        # 🔥 Step 2: Retrieval (UPDATED SIGNATURE)
        docs = retrieve_documents(
            query,
            query_type,
            doc_retriever,
            code_retriever,
            debug_retriever
        )

        # 🔥 Step 3: Filter noise
        docs = filter_docs(docs)

        print(f"📊 Retrieved {len(docs)} strong chunks")

        if not docs:
            print("\n⚠️ Answer:")
            print("I don't have enough information to answer that.")
            continue

        # 🔥 Step 4: Build context
        context = "\n\n".join([doc.page_content for doc in docs])

        # 🔥 Step 5: Prompt
        prompt = build_prompt(context, query, query_type)

        # 🔥 Step 6: Generate response
        response = llm.invoke(prompt)

        print("\n💡 Answer:")
        print(response)

        print("\n📚 Sources:")
        for i, doc in enumerate(docs):
            print(f"\nSource {i+1}:")
            print(doc.metadata.get("source", "Unknown source"))


if __name__ == "__main__":
    run_chat()