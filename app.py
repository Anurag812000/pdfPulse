import streamlit as st
from backend.extract import extract_text_from_pdf, split_text
from backend.pinecone import initialize_pinecone, store_in_pinecone, query_pinecone
from backend.generate_response import (
    initialize_gemini,
    generate_embeddings,
    generate_response,
)
from pinecone.core.openapi.shared.exceptions import UnauthorizedException

from log import logger

# Page Configuration
st.set_page_config(
    page_title="Question Answering System",
    layout="centered",
    page_icon="😈",
)


# Sidebar: API Key Inputs
st.sidebar.title(":green[API Configuration]")
GEMINI_API = st.sidebar.text_input(
    ":blue[Gemini API Key]",
    type="password",
    help="Enter your Gemini API key.",
)
PINECONE_API = st.sidebar.text_input(
    ":violet[Pinecone API Key]", type="password", help="Enter your Pinecone API key."
)

if GEMINI_API and PINECONE_API:
    try:
        initialize_gemini(GEMINI_API)
        index_name = "rag-qa-index"
        index = initialize_pinecone(PINECONE_API, index_name)
        st.sidebar.success("API keys configured successfully!")
    except UnauthorizedException as e:
        st.sidebar.error(f"{e.body}")
        logger.info(e.body)
else:
    st.sidebar.warning("Please enter your API keys.")

st.sidebar.markdown(":red[_Your API Keys are used locally and is NOT saved._]")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload your PDF file", type=["pdf"], help="Upload a PDF document for processing."
)
st.markdown(":gray[_Larger files may take longer to process._]")
if uploaded_file:
    with st.spinner("Processing the file..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        embeddings = generate_embeddings(chunks)
        store_in_pinecone(index, chunks, embeddings)  # type: ignore
        st.success("File processed and indexed successfully!")


# Chat
st.header("Chat with the RAG QA System")
chat_history = st.container()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"],
        avatar=(
            ":material/robot_2:"
            if message["role"] == "ai" or "assistant"
            else ":material/person:"
        ),
    ):
        st.markdown(f"{message['content']}")


# Query input
if query := st.chat_input("Ask a question..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.spinner("Generating response..."):
        query_embedding = generate_embeddings([query])[0]
        relevant_chunks = query_pinecone(index, query_embedding, top_k=5)  # type: ignore
        context = "\n".join(relevant_chunks)
        ai_response = generate_response(query, context)
    st.session_state["messages"].append({"role": "assistant", "content": ai_response})

    # Refresh chat display
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Footer
st.sidebar.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/leeh-nix/PDF-Chatbot)
    """,
    unsafe_allow_html=True,
)
