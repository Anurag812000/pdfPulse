import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from log import logger
import streamlit as st


# @st.cache_data
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    logger.info("extracting text from pdf...")
    return text


# @st.cache_data
def split_text(text, chunk_size=1000, chunk_overlap=200):
    logger.info("splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)
    # logger.info(f"CHUNKS: {chunks}")
    return chunks
