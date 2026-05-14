import streamlit as st
from agents.a1 import process_query
from evaluation.evaluator import evaluate_live_query, final_score
from PIL import Image
import tempfile

st.set_page_config(page_title="AI Debugging Assistant", layout="wide")

st.title("🚀 AI Debugging Assistant")
st.write("Analyze code, debug errors, or understand screenshots.")


# ------------------------------
# SESSION STATE
# ------------------------------
if "text_result" not in st.session_state:
    st.session_state.text_result = None

if "text_query_saved" not in st.session_state:
    st.session_state.text_query_saved = None

if "image_result" not in st.session_state:
    st.session_state.image_result = None

if "image_query_saved" not in st.session_state:
    st.session_state.image_query_saved = None


# ==================================================
# TEXT QUERY SECTION
# ==================================================
st.subheader("💬 Ask a Question")

query = st.text_input("Enter your question")

if st.button("Get Answer", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = process_query(query)

        st.session_state.text_query_saved = query
        st.session_state.text_result = result


# ------------------------------
# SHOW LAST RESULT (TEXT / IMAGE)
# ------------------------------
if st.session_state.text_result:
    result = st.session_state.text_result
    
    st.success(result["answer"])
    st.caption(f"🧠 Type: {result['type']} | ⏱ Time: {result['time']} sec")

    if result.get("sources"):
        with st.expander("📚 Sources"):
            for src in result["sources"]:
                st.write(f"- {src}")

    # 🔥 Evaluate same response
    if st.button("📊 Evaluate This Response", use_container_width=True):

        scores = final_score(
            st.session_state.text_query_saved,
            result["answer"],
            result.get("sources", []),
            result.get("type", "general")
        )

        st.subheader("📈 Evaluation Dashboard")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Relevance", scores["Relevance"])
        c2.metric("Grounded", scores["Groundedness"])
        c3.metric("Complete", scores["Completeness"])
        c4.metric("Structure", scores["Structure"])
        c5.metric("Final", scores["Final Score"])


# ==================================================
# IMAGE QUERY SECTION
# ==================================================
st.subheader("🖼 Analyze Screenshot")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"]
)

image_query = st.text_input(
    "Ask something about this image",
    key="image_query"
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Screenshot",
        width=450
    )

    if st.button("Analyze Image", use_container_width=True):

        if not image_query.strip():
            st.warning("Please ask a question about the image.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            with st.spinner("Analyzing screenshot..."):
                result = process_query(
                    temp_path,
                    is_image=True,
                    image_question=image_query
                )

            # store session
            st.session_state.image_query_saved = image_query
            st.session_state.image_result = result


# ==================================================
# SHOW IMAGE RESULT (FIXED)
# ==================================================
if st.session_state.image_result:

    result = st.session_state.image_result

    st.subheader("🧠 Analysis Result")

    # ------------------ WRAPPED ANSWER ------------------
    st.markdown(
        f"""
<div style="
white-space: pre-wrap;
word-wrap: break-word;
overflow-wrap: break-word;
padding: 14px;
border-radius: 10px;
background-color: #1e1e1e;
color: #ffffff;
line-height: 1.6;
font-size: 15px;
">
{result["answer"]}
</div>
""",
        unsafe_allow_html=True
    )

    st.caption(
        f"🖼 Type: {result['type']} | ⏱ Time: {result['time']} sec"
    )

    # ------------------ OCR TEXT ------------------
    st.subheader("🔍 Extracted Text")

    if result.get("ocr_text"):
        st.text_area(
            label="OCR Output",
            value=result["ocr_text"],
            height=180
        )
    else:
        st.warning("No text detected in image.")

    # ------------------ SOURCES (FIXED MISSING PART) ------------------
    if result.get("sources"):
        with st.expander("📚 Sources"):
            for src in result["sources"]:
                st.write(f"- {src}")

    # ------------------ EVALUATION BUTTON (FIXED MISSING PART) ------------------
    if st.button("📊 Evaluate This Image Response", use_container_width=True):

        scores = final_score(
            st.session_state.image_query_saved,
            result["answer"],
            result.get("sources", []),
            result.get("type", "general")
        )

        st.subheader("📈 Evaluation Dashboard")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Relevance", scores["Relevance"])
        c2.metric("Grounded", scores["Groundedness"])
        c3.metric("Complete", scores["Completeness"])
        c4.metric("Structure", scores["Structure"])
        c5.metric("Final", scores["Final Score"])