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
import asyncio

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details. If the answer is not in the provided context,
    just say "answer is not available in the context". Don't provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = prompt | model | StrOutputParser()
    return chain


def user_input(user_question, pdf_docs, conversation_history):
    if not pdf_docs:
        st.warning("Please upload PDF files before asking a question.")
        return

    with st.spinner("Processing your question..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        context = "\n\n".join([doc.page_content for doc in docs])
        chain = get_conversational_chain()
        response_text = chain.invoke({"context": context, "question": user_question})

    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    conversation_history.append((
        user_question,
        response_text,
        "Google AI",
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ", ".join(pdf_names)
    ))

    st.markdown(
        """
        <style>
            .chat-message {
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }
            .chat-message.user { background-color: #2b313e; }
            .chat-message.bot  { background-color: #475063; }
            .chat-message .avatar { width: 20%; }
            .chat-message .avatar img {
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }
            .chat-message .message {
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # for q, a, model, ts, pdf_name in reversed(conversation_history):
    #     st.markdown(
    #         f"""
    #         <div style="
    #             background-color: #1e2130;
    #             border-radius: 12px;
    #             padding: 1.2rem 1.5rem;
    #             margin-bottom: 1rem;
    #             border: 1px solid #3a3f55;
    #         ">
    #             <div style="display:flex; align-items:center; margin-bottom:0.8rem;">
    #                 <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"
    #                      style="width:36px; height:36px; border-radius:50%; margin-right:10px;">
    #                 <span style="color:#ffffff; font-weight:600;">You</span>
    #             </div>
    #             <p style="color:#d0d3e0; margin:0 0 1rem 46px;">{q}</p>
    #           <p style="color:{'#f0a500' if 'not available' in {a} else '#d0d3e0'}; margin:0 0 0 46px;">{a}</p>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )


    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Study Bot", page_icon=":books:")
    st.header("Study Bot :books:")

    if not api_key:
        st.error("Google API key not found. Please add GOOGLE_API_KEY to your .env file.")
        return

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    linkedin_profile_link = "https://www.linkedin.com/in/shakthinr/"
    github_profile_link   = "https://github.com/ShakthiNR/"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    with st.sidebar:
        st.title("Study Bot")
        st.markdown("<p style='font-size:13px; color:gray;'>Powered by Gemini AI. Upload your documents below to begin an intelligent conversation with your PDF content.</p>", unsafe_allow_html=True)
        st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

        pdf_docs = st.file_uploader(
            "Upload here",
            #"Upload your PDF Files and Click Submit & Process",
            accept_multiple_files=True
        )

        col1, col2 = st.columns(2)
        submit_button = col1.button("Process")
        reset_button  = col2.button("Reset")

        if submit_button:
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files before processing.")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = ""
            st.rerun()

    if 'processed' not in st.session_state:
        st.session_state.processed = False

    user_question = st.text_input(
        "Ask a Question from the PDF Files",
        key="input"
    )

    # if user_question and not st.session_state.processed:
    #     st.session_state.processed = True
    #     user_input(
    #         user_question,
    #         pdf_docs if 'pdf_docs' in dir() else None,
    #         st.session_state.conversation_history
    #     )

    # if not user_question:
    #     st.session_state.processed = False
    if user_question and not st.session_state.processed:
        st.session_state.processed = True
        user_input(
            user_question,
            pdf_docs if 'pdf_docs' in dir() else None,
            st.session_state.conversation_history
        )
        st.session_state.last_question = user_question

    if user_question and user_question != st.session_state.get('last_question', ''):
        st.session_state.processed = False

    if not user_question:
        st.session_state.processed = False

    # Always render full conversation history
    if st.session_state.conversation_history:
        for q, a, model, ts, pdf_name in reversed(st.session_state.conversation_history):
            st.markdown(
                f"""
                <div style="
                    background-color: #1e2130;
                    border-radius: 12px;
                    padding: 1.2rem 1.5rem;
                    margin-bottom: 1rem;
                    border: 1px solid #3a3f55;
                ">
                    <div style="display:flex; align-items:center; margin-bottom:0.8rem;">
                        <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"
                            style="width:36px; height:36px; border-radius:50%; margin-right:10px;">
                        <span style="color:#ffffff; font-weight:600;">You</span>
                    </div>
                    <p style="color:#d0d3e0; margin:0 0 1rem 46px;">{q}</p>
                    <hr style="border:none; border-top:1px solid #3a3f55; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; margin-bottom:0.8rem;">
                        <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"
                            style="width:36px; height:36px; border-radius:50%; margin-right:10px;">
                        <span style="color:#ffffff; font-weight:600;">Assistant</span>
                    </div>
                    <p style="color:#d0d3e0; margin:0 0 0 46px;">{a}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()