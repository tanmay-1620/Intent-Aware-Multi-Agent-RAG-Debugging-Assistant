from rag.retriever import create_rag_system
from multimodals.image_reader import extract_text_from_image
import time


# 🔥 Query classification
def classify_query(query):
    q = query.lower()

    if any(word in q for word in ["error", "fail", "bug", "issue", "not working", "500", "401"]):
        return "debug"

    elif any(word in q for word in ["how", "flow", "process", "working"]):
        return "flow"

    elif any(word in q for word in ["what is", "define", "meaning"]):
        return "definition"

    elif any(word in q for word in ["api", "route", "endpoint", "code", "file", "function", "where"]):
        return "code"

    else:
        return "general"


# 🔥 Retrieval strategy
def retrieve_docs(query, query_type, doc_retriever, code_retriever, debug_retriever):
    if query_type == "code":
        return code_retriever.invoke(query)

    elif query_type == "flow":
        return code_retriever.invoke(query) + doc_retriever.invoke(query)

    elif query_type == "definition":
        return doc_retriever.invoke(query)

    elif query_type == "debug":
        return debug_retriever.invoke(query)

    else:
        return doc_retriever.invoke(query)


# 🔥 Filter weak chunks
def filter_docs(docs):
    return [doc for doc in docs if len(doc.page_content.strip()) > 80]


# 🔥 Prompt builder
def build_prompt(context, query, query_type):
    if query_type == "debug":
        return f"""
You are an AI debugging assistant.

Use ONLY the context.

STRICT RULES:
- Use ONLY context
- Keep answer under 120 words
- No long paragraphs
- Use short bullet points
- Be clear and direct

Format:

🔍 Summary:
...

🧠 Causes:
1. ...
2. ...

🛠 Fixes:
1. ...
2. ...

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
You are an AI assistant specialized in understanding codebases and technical documentation.

STRICT RULES:
- Answer ONLY using the provided context
- DO NOT guess
- If answer is not present, say:
  "I don't have enough information to answer that."

Behavior:
- If code-related → explain code clearly
- If flow-related → explain step-by-step
- Keep answer structured

Context:
{context}

Question:
{query}

Answer:
"""


def main():
    doc_retriever, code_retriever, debug_retriever, llm = create_rag_system()

    print("\n🚀 Intelligent AI Debugging Assistant Ready!")
    print("Type 'exit' to quit\n")

    while True:
        start = time.time()

        # 🔥 Reset per iteration
        image_mode = False
        image_context = ""

        raw_query = input("💬 Ask your question: ")

        # 🔥 Exit FIRST
        if raw_query.lower() == "exit":
            print("\n👋 Exiting... Goodbye!")
            break

        query = raw_query

        # 🔥 Image handling
        if raw_query.lower().startswith("image:"):
            image_path = raw_query.replace("image:", "").strip()
            extracted = extract_text_from_image(image_path)

            print("\n🖼 Extracted Text:")
            print(extracted)

            image_mode = True
            image_context = extracted

            query = f"Analyze this screenshot issue:\n{extracted}"

        # 🔥 Step 1: Classification
        query_type = classify_query(query)
        print(f"\n🧠 AI Detected Type: {query_type}")

        # 🔥 Step 2–6: Retrieval OR Image Context
        if image_mode:
            docs = []
            context = image_context
            print("📊 Using image text as context")
        else:
            docs = retrieve_docs(query, query_type, doc_retriever, code_retriever, debug_retriever)
            docs = filter_docs(docs)

            print(f"📊 Retrieved {len(docs)} strong chunks")

            if not docs:
                print("\n⚠️ Answer:")
                print("I don't have enough information to answer that.")
                continue

            print("\n🔍 Retrieved Context:")
            for i, doc in enumerate(docs):
                print(f"\n--- Chunk {i+1} ({doc.metadata.get('type')}) ---")
                print(doc.page_content[:200])

            context = "\n\n".join([doc.page_content[:300] for doc in docs[:2]])

        # 🔥 Step 7: Prompt
        prompt = build_prompt(context, query, query_type)

        # 🔥 Step 8: LLM response
        response = llm.invoke(prompt)

        print("\n💡 Answer:")
        print(response if isinstance(response, str) else getattr(response, "content", response))

        # 🔥 Step 9: Sources (skip for image mode)
        if not image_mode:
            print("\n📚 Sources:")
            for i, doc in enumerate(docs):
                print(f"\nSource {i+1}:")
                print(doc.metadata.get("source", "Unknown source"))

        # ✅ Time
        print(f"\n⏱ Total Time Taken: {round(time.time() - start, 2)} sec")


if __name__ == "__main__":
    main()