from pinecone import Pinecone, ServerlessSpec


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
