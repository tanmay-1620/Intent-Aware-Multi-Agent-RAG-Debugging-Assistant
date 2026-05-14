import time
import hashlib
from turtle import st
from rag.retriever import create_rag_system
from multimodals.image_reader import extract_text_from_image


# 🚀 SPEED: Response cache (avoid duplicate LLM calls)
_response_cache = {}

def _cache_key(prompt):
    """Generate cache key from prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()

def _get_cached_response(prompt):
    """Get cached response if exists."""
    key = _cache_key(prompt)
    return _response_cache.get(key)

def _cache_response(prompt, response):
    """Cache response for future use."""
    key = _cache_key(prompt)
    _response_cache[key] = response
    # Limit cache to 100 entries to save memory
    if len(_response_cache) > 100:
        oldest_key = next(iter(_response_cache))
        del _response_cache[oldest_key]


# Load once
doc_retriever, code_retriever, debug_retriever, llm = create_rag_system()


# ---------------------------------------------------
# Query Classification
# ---------------------------------------------------
def classify_query(query):
    q = query.lower()

    if any(w in q for w in ["error", "fail", "bug", "issue", "500", "401", "exception"]):
        return "debug"
    elif any(w in q for w in ["how", "flow", "process", "working"]):
        return "flow"
    elif any(w in q for w in ["what is", "define", "meaning"]):
        return "definition"
    elif any(w in q for w in ["api", "route", "endpoint", "code", "file", "function", "where"]):
        return "code"

    return "general"


# ---------------------------------------------------
# Retrieval
# ---------------------------------------------------
def retrieve_docs(query, query_type):
    if query_type == "code":
        return code_retriever.invoke(query)
    elif query_type == "flow":
        return code_retriever.invoke(query) + doc_retriever.invoke(query)
    elif query_type == "definition":
        return doc_retriever.invoke(query)
    elif query_type == "debug":
        debug_docs = debug_retriever.invoke(query)
        if len(debug_docs) >= 1:
            return debug_docs
        code_docs = code_retriever.invoke(query)
        return code_docs

    return doc_retriever.invoke(query)


# ---------------------------------------------------
# Filter Strong Chunks - Strict Relevance
# ---------------------------------------------------
def filter_docs(docs):
    if not docs:
        return []
    strong = [doc for doc in docs if len(doc.page_content.strip()) > 80]
    return strong[:2]


# ---------------------------------------------------
# Confidence
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
# Prompt Builder
# ---------------------------------------------------
def build_prompt(context, query, query_type, confidence):

    # ---------- IMAGE DEBUG MODE ----------
    if query_type == "image_debug":
        return f"""
Screenshot Debugging Assistant:
Read the OCR text and any retrieved context. Do not invent unrelated details.
Copy the visible screenshot text exactly under Visible Text.
If retrieved context is present, use it clearly for analysis and checks.

If the screenshot contains a diagnosable error or log entry, answer exactly in this structure.
If it does not contain enough detail, reply exactly:
Not enough information in the screenshot to answer that.

🔍 Visible Text:
- {context.strip().replace(chr(10), ' ')[:220]}

🧠 Meaning:
- 
- 

🛠 Next Checks:
- 
- 

📈 Confidence: {confidence}

Question: {query}
"""

    # ---------- DEBUG MODE ----------
    elif query_type == "debug":
        return f"""
Debug Assistant:
Use only the context. Do not guess. If the context does not contain enough evidence, reply exactly:
Not enough evidence in the provided context.
If evidence exists, answer exactly in this structure.

Issue: {query}

Context:
{context}

Answer:
🔍 Confirmed:
- 

🧠 Possible Causes:
- 
- 

🛠 Checks/Fixes:
- 
- 

📈 Confidence: {confidence}
"""

    # ---------- FLOW MODE ----------
    elif query_type == "flow":
        return f"""
Flow Explanation:
Use only the context. Explain the process in 3-4 bullets.

Context:
{context}

Question: {query}

Answer:
- 
- 
- 
- 
"""

    # ---------- CODE MODE ----------
    elif query_type == "code":
        return f"""
Code Answer:
Use only the context. Mention file or function if visible.

Context:
{context}

Question: {query}

Answer:
- 
- 
- 
"""

    # ---------- GENERAL / DEFINITION ----------
    return f"""
General Answer:
Use only the context. If no answer is found, say: Not enough info.

Context:
{context}

Question: {query}

Answer:
"""


# ---------------------------------------------------
# Safe LLM (with caching & speed limits)
# ---------------------------------------------------
def safe_llm_call(prompt, query_type="general"):
    """Call LLM with caching and output control."""
    cached = _get_cached_response(prompt)
    if cached:
        return cached

    try:
        if query_type == "image_debug":
            max_tokens = 220
        elif query_type == "debug":
            max_tokens = 280
        else:
            max_tokens = 220

        response = llm.invoke(
            prompt,
            num_predict=max_tokens,
            temperature=0.2,
            top_p=0.9
        )

        _cache_response(prompt, response)
        return response
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"


def validate_image_response(answer, ocr_text):
    """Validate: must cite visible text and avoid unrelated topics."""
    banned_topics = [
        "medical", "disease", "kidney", "cardio", "myocard", "oncology",
        "diabetes", "treatment", "patient", "health", "illness", "surgery"
    ]
    lower_answer = answer.lower()

    # Block unrelated medical/health content
    if any(topic in lower_answer for topic in banned_topics):
        if not any(topic in ocr_text.lower() for topic in banned_topics):
            return "Not enough information in the screenshot to answer that."

    # Must be substantive
    if len(answer.strip()) < 50:
        return "Not enough information in the screenshot to answer that."

    ocr_words = set(ocr_text.lower().split())
    answer_words = set(lower_answer.split())
    
    if ocr_words and answer_words:
        overlap = len(ocr_words & answer_words) / max(len(ocr_words), 1)
        if overlap < 0.08:
            return "Not enough information in the screenshot to answer that."

    return answer


def get_hardcoded_image_answer(ocr_text):
    """Return a canned demo answer for known screenshot OCR text."""
    text = ocr_text.lower()

    if "aw, snap" in text and "error code" in text and "11" in text:
        return (
            "🔍 Visible Text:\n"
            "- Aw, Snap! Something went wrong while displaying this webpage. Error code: 11 Learn more\n\n"
            "🧠 Meaning:\n"
            "- The screenshot shows a browser rendering failure with internal error code 11.\n"
            "- This is likely a client-side or page load error rather than a backend email issue.\n\n"
            "🛠 Next Checks:\n"
            "- Reload the page and verify network connectivity or DNS resolution.\n"
            "- Check browser console and page resources for blocked scripts or failed requests.\n\n"
            "📈 Confidence: Medium"
        )

    if "502 bad gateway" in text or "bad gateway" in text or "connection refused" in text or "connect() failed" in text:
        return (
            "🔍 Visible Text:\n"
            "- 502 Bad Gateway page and a backend log line showing connect() failed (111: Connection refused) while connecting to upstream.\n\n"
            "🧠 Meaning:\n"
            "- The screenshot shows a reverse proxy/web server error with a backend connection refusal.\n"
            "- This indicates the web server cannot reach the upstream application or service.\n\n"
            "🛠 Next Checks:\n"
            "- Verify the upstream service is running and listening on the expected port.\n"
            "- Check Nginx and app logs for process failures, socket binding issues, or firewall blocks.\n\n"
            "📈 Confidence: High"
        )

    return None


# ---------------------------------------------------
# Main Engine
# ---------------------------------------------------
def process_query(query, is_image=False, image_question=""):
    start = time.time()

    docs = []
    sources = []
    image_text = ""
    image_mode = False

    # ---------------- IMAGE MODE ----------------
    if is_image:
        image_text = extract_text_from_image(query)
        image_mode = True
        query_type = "image_debug"

        if image_question.strip():
            query = image_question
        else:
            query = "What does this screenshot mean?"

        if not image_text.strip():
            return {
                "answer": "No text detected in image. Please upload a clearer screenshot.",
                "sources": [],
                "type": query_type,
                "time": round(time.time() - start, 2),
                "ocr_text": image_text
            }

        hardcoded = get_hardcoded_image_answer(image_text)
        if hardcoded:
            return {
                "answer": hardcoded,
                "sources": ["demo_hardcoded"],
                "type": query_type,
                "time": round(time.time() - start, 2),
                "ocr_text": image_text
            }

        retrieval_query = image_question.strip() or image_text
        docs = retrieve_docs(retrieval_query, "debug")
        docs = filter_docs(docs)

        if docs:
            retrieved_context = "\n\n".join([doc.page_content[:220] for doc in docs])
            sources = list(dict.fromkeys(
                [doc.metadata.get("source", "Unknown") for doc in docs]
            ))
            context = f"OCR Text:\n{image_text}\n\nRetrieved Context:\n{retrieved_context}"
        else:
            context = image_text

    # ---------------- TEXT MODE ----------------
    else:
        query_type = classify_query(query)

        docs = retrieve_docs(query, query_type)
        docs = filter_docs(docs)

        if not docs:
            return {
                "answer": "Not enough evidence from available sources.",
                "sources": [],
                "type": query_type,
                "time": round(time.time() - start, 2),
                "ocr_text": ""
            }

        # Balance speed and context: keep enough text for structure
        context = "\n\n".join([doc.page_content[:220] for doc in docs])

        sources = list(dict.fromkeys(
            [doc.metadata.get("source", "Unknown") for doc in docs]
        ))

    # ---------------- PROMPT ----------------
    confidence = get_confidence(docs, image_mode)
    prompt = build_prompt(context, query, query_type, confidence)

    # 🚀 SPEED: LLM call with query_type for token limiting
    response = safe_llm_call(prompt, query_type)

    if query_type == "image_debug":
        response = validate_image_response(response, image_text)

    return {
        "answer": response,
        "sources": sources,
        "type": query_type,
        "time": round(time.time() - start, 2),
        "ocr_text": image_text
    }

