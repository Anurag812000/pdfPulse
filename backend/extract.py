import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(uploaded_file):
    print("extracting text from pdf...")
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def split_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
