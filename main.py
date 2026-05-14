from rag.retriever import create_rag_system
from multimodals.image_reader import extract_text_from_image
import time


# ---------------------------------------------------
# FAST Query Classification (No LLM delay)
# ---------------------------------------------------
def classify_query(query):
    q = query.lower()

    if any(word in q for word in ["error", "fail", "bug", "issue", "500", "401"]):
        return "debug"

    elif any(word in q for word in ["how", "flow", "process", "working"]):
        return "flow"

    elif any(word in q for word in ["what is", "define", "meaning"]):
        return "definition"

    elif any(word in q for word in ["api", "route", "endpoint", "code", "file", "function", "where"]):
        return "code"

    return "general"


# ---------------------------------------------------
# Retrieval Routing
# ---------------------------------------------------
def retrieve_docs(query, query_type, doc_retriever, code_retriever, debug_retriever):
    if query_type == "code":
        return code_retriever.invoke(query)

    elif query_type == "flow":
        return code_retriever.invoke(query) + doc_retriever.invoke(query)

    elif query_type == "definition":
        return doc_retriever.invoke(query)

    elif query_type == "debug":
        return debug_retriever.invoke(query)

    return doc_retriever.invoke(query)


# ---------------------------------------------------
# Strong Chunk Filtering (reduces garbage context)
# ---------------------------------------------------
def filter_docs(docs):
    strong_docs = []

    for doc in docs:
        text = doc.page_content.strip()

        if len(text) > 80 and "..." not in text:
            strong_docs.append(doc)

    return strong_docs[:2]   # 🔥 keep small for speed


# ---------------------------------------------------
# Confidence Logic
# ---------------------------------------------------
def get_confidence(docs, image_mode=False):
    if image_mode:
        return "Medium"

    if len(docs) == 0:
        return "Low"
    elif len(docs) == 1:
        return "Medium"

    return "High"


# ---------------------------------------------------
# Prompt Builder (STRICT + FAST)
# ---------------------------------------------------
def build_prompt(context, query, query_type, confidence):
    if query_type == "debug":
        return f"""
You are an AI debugging assistant.

STRICT RULES:
1. Use ONLY the provided context.
2. Do NOT assume anything outside context.
3. If evidence is missing → say "Not enough evidence".
4. Keep answer SHORT and precise.

Format:

🔍 Confirmed:
...

🧠 Possible Causes:
- ...

🛠 Fix:
- ...

📈 Confidence: {confidence}

Context:
{context}

Issue:
{query}

Answer:
"""

    return f"""
Answer ONLY from context.
If not found → say "I don't have enough information".

Context:
{context}

Question:
{query}

Answer:
"""


# ---------------------------------------------------
# Safe LLM Call (prevents hanging)
# ---------------------------------------------------
def safe_llm_call(llm, prompt):
    try:
        return llm.invoke(prompt)
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"


# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------
def main():
    doc_retriever, code_retriever, debug_retriever, llm = create_rag_system()

    print("\n🚀 AI Debugging Assistant Ready!")
    print("Type 'exit' to quit\n")

    while True:
        start = time.time()

        image_mode = False
        image_context = ""
        docs = []

        query = input("💬 Ask your question: ")

        if query.lower() == "exit":
            print("\n👋 Exiting...")
            break

        # ---------------------------------------------------
        # IMAGE MODE
        # ---------------------------------------------------
        if query.lower().startswith("image:"):
            image_path = query.replace("image:", "").strip()

            extracted = extract_text_from_image(image_path)

            print("\n🖼 Extracted Text:")
            print(extracted)

            image_mode = True
            image_context = extracted
            query = f"Analyze this error:\n{extracted}"

        # ---------------------------------------------------
        # CLASSIFICATION (FAST)
        # ---------------------------------------------------
        query_type = classify_query(query)
        print(f"🧠 Type: {query_type}")

        # ---------------------------------------------------
        # CONTEXT
        # ---------------------------------------------------
        if image_mode:
            context = image_context

        else:
            docs = retrieve_docs(query, query_type, doc_retriever, code_retriever, debug_retriever)
            docs = filter_docs(docs)

            print(f"📊 Chunks: {len(docs)}")

            if not docs:
                print("\n⚠️ Not enough evidence.")
                continue

            context = "\n\n".join([doc.page_content[:250] for doc in docs])

        # ---------------------------------------------------
        # PROMPT
        # ---------------------------------------------------
        confidence = get_confidence(docs, image_mode)
        prompt = build_prompt(context, query, query_type, confidence)

        # ---------------------------------------------------
        # LLM (MAIN DELAY HERE)
        # ---------------------------------------------------
        response = safe_llm_call(llm, prompt)

        # ---------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------
        print("\n💡 Answer:")
        print(response)

        # ---------------------------------------------------
        # SOURCES
        # ---------------------------------------------------
        if docs:
            print("\n📚 Sources:")
            shown = set()

            for doc in docs:
                src = doc.metadata.get("source", "Unknown")
                if src not in shown:
                    print("-", src)
                    shown.add(src)

        # ---------------------------------------------------
        # TIME
        # ---------------------------------------------------
        print(f"\n⏱ Time: {round(time.time() - start, 2)} sec\n")


# ---------------------------------------------------
if __name__ == "__main__":
    main()