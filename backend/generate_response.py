import google.generativeai as genai
from log import logger


def initialize_gemini(api_key):
    genai.configure(api_key=api_key)


# @st.cache_data
def generate_embeddings(content_list, model="models/embedding-001"):
    logger.info("Generating embeddings...")
    # logger.info(f"Content list: {content_list}")
    embeddings = []
    for content in content_list:
        result = genai.embed_content(
            model=model, content=content, task_type="retrieval_document"
        )
        # Explicitly convert to list to ensure compatibility
        embeddings.append(list(result["embedding"]))

    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings


def generate_response(query, context, model_name="gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    prompt = f"Role: Use the provided context to answer the question. Stick to the points given in the context but summarize appropriately. \n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    logger.info("generating response...")
    logger.info(f"CONTEXT: {context}\n\n\nQUERY:{query}")
    response = model.generate_content(prompt)
    return response.text
