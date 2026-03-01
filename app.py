import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def get_pdf_text(pdf_docs):
    """Extract raw text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    """Split raw text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


def get_embeddings():
    """Return a HuggingFace sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(text_chunks):
    """Embed text chunks and save a FAISS index to disk."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def load_vector_store():
    """Load the previously saved FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )


def get_conversational_chain():
    """Build a simple prompt → LLM → string-output chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details. If the answer is not in the provided context,
    just say "Answer is not available in the context". Don't provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    return prompt | model | StrOutputParser()


# ─────────────────────────────────────────────
# Core Q&A function  (no re-processing PDFs)
# ─────────────────────────────────────────────

def answer_question(user_question, pdf_names):
    """
    Load the pre-built FAISS index, retrieve relevant chunks,
    call Gemini, append result to conversation history.
    """
    with st.spinner("Thinking…"):
        db = load_vector_store()
        docs = db.similarity_search(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])

        chain = get_conversational_chain()
        response_text = chain.invoke({"context": context, "question": user_question})

    st.session_state.conversation_history.append((
        user_question,
        response_text,
        "Google AI",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ", ".join(pdf_names),
    ))


# ─────────────────────────────────────────────
# Sidebar CSV download helper
# ─────────────────────────────────────────────

def render_csv_download():
    if st.session_state.conversation_history:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"],
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = (
            f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">'
            f"<button>⬇ Download conversation history as CSV</button></a>"
        )
        st.sidebar.markdown(href, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Conversation renderer
# ─────────────────────────────────────────────

CHAT_CSS = """
<style>
.chat-bubble {
    background-color: #1e2130;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #3a3f55;
}
.chat-bubble .avatar-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.8rem;
}
.chat-bubble .avatar-row img {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin-right: 10px;
}
.chat-bubble .avatar-row span {
    color: #ffffff;
    font-weight: 600;
}
.chat-bubble hr {
    border: none;
    border-top: 1px solid #3a3f55;
    margin: 0.8rem 0;
}
.chat-bubble p {
    color: #d0d3e0;
    margin: 0 0 0 46px;
}
</style>
"""

USER_ICON = "https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"
BOT_ICON  = "https://i.ibb.co/wNmYHsx/langchain-logo.webp"


def render_conversation():
    if not st.session_state.conversation_history:
        return

    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    for q, a, model, ts, pdf_name in reversed(st.session_state.conversation_history):
        st.markdown(
            f"""
            <div class="chat-bubble">
                <div class="avatar-row">
                    <img src="{USER_ICON}">
                    <span>You</span>
                    <span style="color:#888; font-size:12px; margin-left:auto;">{ts}</span>
                </div>
                <p>{q}</p>
                <hr>
                <div class="avatar-row">
                    <img src="{BOT_ICON}">
                    <span>Assistant</span>
                </div>
                <p>{a}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Study Bot", page_icon=":books:")
    st.header("Study Bot :books:")

    # ── Guard: API key ──
    if not api_key:
        st.error("Google API key not found. Please add GOOGLE_API_KEY to your .env file.")
        return

    # ── Session-state initialisation ──
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "pdf_names" not in st.session_state:
        st.session_state.pdf_names = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    # ── Sidebar ──
    linkedin_profile_link = "https://www.linkedin.com/in/username/"
    github_profile_link   = "https://github.com/username/"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    with st.sidebar:
        st.title("Study Bot")
        st.markdown(
            "<p style='font-size:13px; color:gray;'>"
            "Powered by Gemini AI. Upload your documents below to begin an "
            "intelligent conversation with your PDF content.</p>",
            unsafe_allow_html=True,
        )

        # Full-width buttons via CSS
        st.markdown(
            "<style>div.stButton > button { width: 100%; }</style>",
            unsafe_allow_html=True,
        )

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"],
        )

        col1, col2 = st.columns(2)
        submit_button = col1.button("Process")
        reset_button  = col2.button("Reset")

        # ── Process button ──
        if submit_button:
            if pdf_docs:
                with st.spinner("Processing PDFs…"):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("Could not extract text from the uploaded PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        build_vector_store(text_chunks)
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_names = [p.name for p in pdf_docs]
                        st.success("PDFs processed successfully! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file before processing.")

        # ── Reset button ──
        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.last_question = ""
            st.session_state.pdf_processed = False
            st.session_state.pdf_names = []
            st.rerun()

        # ── CSV download ──
        # render_csv_download()

    # ── Main area: question input ──
    user_question = st.text_input(
        "Ask a question from the PDF files",
        key="input",
        placeholder="Type your question here and press Enter…",
    )

    if user_question:
        if not st.session_state.pdf_processed:
            st.warning("Please upload and process PDF files first (use the sidebar).")
        elif user_question != st.session_state.last_question:
            # New question → call the LLM
            answer_question(user_question, st.session_state.pdf_names)
            st.session_state.last_question = user_question

    # ── Render full conversation ──
    render_conversation()


if __name__ == "__main__":
    main()