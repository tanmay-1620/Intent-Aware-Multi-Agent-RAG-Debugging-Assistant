import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Lazy import to avoid loading vector store on module import
_process_query = None

def get_process_query():
    global _process_query
    if _process_query is None:
        from agents.a1 import process_query
        _process_query = process_query
    return _process_query

# Load embedding model for semantic similarity
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            # Fallback if model not available
            _embedding_model = None
    return _embedding_model

def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))

def score_relevance(query, answer, query_type="general"):
    """Improved relevance scoring with semantic similarity for image queries."""

    # For image queries, evaluate based on OCR-style evidence and structure
    if query_type == "image_debug":
        lower = answer.lower()
        if "not enough information" in lower or "no text detected" in lower:
            return 10.0

        score = 20
        if "🔍 visible text:" in lower or "visible text:" in lower or "ocr text" in lower:
            score += 25
        if "error code" in lower or "gateway" in lower or "connection" in lower or "timeout" in lower or "server" in lower or "snap" in lower or "error" in lower or "crash" in lower:
            score += 30
        if "🧠" in lower or "meaning" in lower:
            score += 15
        if "🛠" in lower or "next checks" in lower or "check" in lower or "fix" in lower or "solution" in lower or "verify" in lower:
            score += 15
        
        word_count = len(answer.split())
        if word_count > 40:
            score += 10
        elif word_count > 20:
            score += 5
        
        return min(100.0, score)

    # For text queries, use hybrid approach
    q_tokens = tokenize(query)
    a_tokens = tokenize(answer)

    # Token overlap (original method)
    if q_tokens:
        overlap = len(q_tokens & a_tokens) / len(q_tokens)
        token_score = overlap * 100
    else:
        token_score = 0

    # Semantic similarity (if model available)
    model = get_embedding_model()
    if model:
        try:
            query_emb = model.encode([query])[0]
            answer_emb = model.encode([answer])[0]
            similarity = np.dot(query_emb, answer_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(answer_emb))
            semantic_score = similarity * 100
            # Weighted combination: 40% token, 60% semantic
            return round(0.4 * token_score + 0.6 * semantic_score, 2)
        except:
            pass

    return round(token_score, 2)


def score_completeness(answer):
    """More generous completeness scoring for concise but informative answers."""
    words = len(answer.split())

    # Adjust thresholds for shorter but meaningful responses
    if words < 5:
        return 15  # Very short but still some info
    elif words < 10:
        return 40  # Brief but useful
    elif words < 20:
        return 70  # Good length
    elif words < 40:
        return 90  # Comprehensive
    return 100  # Very detailed


def score_structure(answer):
    """Enhanced structure scoring that recognizes the actual format patterns used."""
    # Primary markers (emojis used in prompts)
    primary_markers = ["🔍", "🧠", "🛠", "📈"]
    primary_count = sum(1 for m in primary_markers if m in answer)

    # Secondary markers (bullets, numbers)
    secondary_markers = ["•", "-", "1.", "2.", "3.", "4."]
    secondary_count = sum(1 for m in secondary_markers if m in answer)

    # Bonus for having multiple sections
    section_bonus = 0
    if primary_count >= 2:
        section_bonus = 20
    elif primary_count >= 1:
        section_bonus = 10

    # Total score: primary markers worth more
    total_score = (primary_count * 15) + (secondary_count * 5) + section_bonus

    return min(total_score, 100)


def score_groundedness(answer, sources, query_type="general"):
    """More nuanced groundedness scoring."""
    if not sources:
        # For image queries, OCR text serves as "source"
        if query_type == "image_debug":
            return 60  # OCR provides grounding
        return 40  # No sources

    # Bonus for multiple sources
    source_bonus = min(len(sources) * 10, 20)
    return min(70 + source_bonus, 100)


def final_score(query, answer, sources, query_type="general"):
    r = score_relevance(query, answer, query_type)
    c = score_completeness(answer)
    s = score_structure(answer)
    g = score_groundedness(answer, sources, query_type)

    total = round(
        (0.25 * r) +
        (0.25 * g) +
        (0.25 * c) +
        (0.25 * s), 2
    )

    return {
        "Relevance": r,
        "Groundedness": g,
        "Completeness": c,
        "Structure": s,
        "Final Score": total
    }


def evaluate_live_query(query, is_image=False, image_question=""):
    process_query = get_process_query()
    result = process_query(
        query=query,
        is_image=is_image,
        image_question=image_question
    )

    # Get query type from result
    query_type = result.get("type", "general")
    relevance_query = query
    if is_image:
        if result.get("ocr_text", "").strip():
            relevance_query = result["ocr_text"].strip()
        elif image_question.strip():
            relevance_query = image_question.strip()

    scores = final_score(
        relevance_query,
        result["answer"],
        result["sources"],
        query_type
    )

    return result, scores


def run_batch_evaluation(queries):
    import pandas as pd

    process_query = get_process_query()
    rows = []

    for q in queries:
        result = process_query(q)
        query_type = result.get("type", "general")
        scores = final_score(q, result["answer"], result["sources"], query_type)

        rows.append({
            "Query": q,
            "Type": result["type"],
            "Score": scores["Final Score"],
            "Time": result["time"]
        })

    return pd.DataFrame(rows)