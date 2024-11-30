import google.generativeai as genai


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
    response = model.generate_content(prompt)
    return response.text
