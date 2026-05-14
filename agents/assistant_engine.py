import time
from rag.retriever import create_rag_system
from multimodals.image_reader import extract_text_from_image

# Load once
doc_retriever, code_retriever, debug_retriever, llm = create_rag_system()


def classify_query(query):
    q = query.lower()

    if any(word in q for word in ["error", "fail", "bug", "issue", "500", "401"]):
        return "debug"
    elif any(word in q for word in ["how", "flow", "process"]):
        return "flow"
    elif any(word in q for word in ["what is", "define", "meaning"]):
        return "definition"
    elif any(word in q for word in ["api", "route", "endpoint", "code", "file", "function", "where"]):
        return "code"
    return "general"


def retrieve_docs(query, query_type):
    if query_type == "code":
        return code_retriever.invoke(query)
    elif query_type == "flow":
        return code_retriever.invoke(query) + doc_retriever.invoke(query)
    elif query_type == "definition":
        return doc_retriever.invoke(query)
    elif query_type == "debug":
        return debug_retriever.invoke(query)
    return doc_retriever.invoke(query)


def filter_docs(docs):
    return [doc for doc in docs if len(doc.page_content.strip()) > 80]


def build_prompt(context, query, query_type):
    if query_type == "debug":
        return f"""
You are an AI debugging assistant.

Use ONLY context.

Rules:
- Keep concise
- Use bullets
- Max 2 causes
- Max 2 fixes

Format:

🔍 Summary:
...

🧠 Causes:
- ...
- ...

🛠 Fixes:
- ...
- ...

📈 Confidence:
Low / Medium / High

Context:
{context}

Issue:
{query}

Answer:
"""
    else:
        return f"""
You are an AI assistant specialized in codebases and documentation.

Rules:
- Use ONLY context
- Keep structured
- Keep concise
- If unknown, say not enough info

Context:
{context}

Question:
{query}

Answer:
"""


def process_query(query, is_image=False):
    start = time.time()

    docs = []
    sources = []
    image_text = ""

    if is_image:
        image_text = extract_text_from_image(query)
        context = image_text
        query = f"Analyze this screenshot issue:\n{image_text}"
        query_type = "debug"

    else:
        query_type = classify_query(query)
        docs = retrieve_docs(query, query_type)
        docs = filter_docs(docs)

        if not docs:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
                "type": query_type,
                "time": round(time.time() - start, 2)
            }

        context = "\n\n".join([doc.page_content[:300] for doc in docs[:2]])
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]

    prompt = build_prompt(context, query, query_type)
    response = llm.invoke(prompt)

    return {
        "answer": response,
        "sources": sources,
        "type": query_type,
        "time": round(time.time() - start, 2),
        "ocr_text": image_text
    }