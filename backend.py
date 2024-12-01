import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai


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


def initialize_pinecone(api_key, index_name, dimension=768):
    pinecone = Pinecone(api_key=api_key)

    if index_name in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(index_name)

    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("indexing...")
    return pinecone.Index(index_name)


def store_in_pinecone(index, chunks, embeddings):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        index.upsert([(str(i), embedding, {"text": chunk})])


def query_pinecone(index, query_embedding, top_k):
    search_results = index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True
    )
    print("querying...")
    return [result["metadata"]["text"] for result in search_results["matches"]]


def initialize_gemini(api_key):
    genai.configure(api_key=api_key)


def generate_embeddings(content_list, model="models/embedding-001"):
    embeddings = []
    for content in content_list:
        result = genai.embed_content(
            model=model, content=content, task_type="retrieval_document"
        )
        embeddings.append(result["embedding"])
    return embeddings


def generate_response(query, context, model_name="gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    prompt = f"Role: Use the provided context to answer the question. Stick to the points given in the context but summarize appropriately. \n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    print("generating response...")
    response = model.generate_content(prompt)
    return response.text
