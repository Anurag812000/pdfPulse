from pinecone import Pinecone, ServerlessSpec
from backend.generate_response import generate_embeddings
from log import logger


# @st.cache_data
def query_pinecone(_index, query_embedding, top_k):
    logger.info("querying...")
    logger.warning(f"\n\nINDEX: {_index}\n\n")
    search_results = _index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True
    )
    for result in search_results["matches"]:
        logger.info(
            f"RESULTS: {round(result['score'], 2)}: {result['metadata']['text']}\n"
        )

    return [result["metadata"]["text"] for result in search_results["matches"]]


def initialize_pinecone(api_key, index_name):
    from pinecone import Pinecone, ServerlessSpec

    if not api_key:
        raise ValueError("Pinecone API key is required")

    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)

        test_embedding = generate_embeddings(["test content"])[0]
        dimension = len(test_embedding)

        logger.info(f"Detected embedding dimension: {dimension}")

        # Check if index exists
        existing_indexes = pc.list_indexes()

        # Create index only if it doesn't exist
        if not any(index.name == index_name for index in existing_indexes):
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            logger.info(f"Index {index_name} already exists")

        return pc.Index(index_name)

    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise


def store_in_pinecone(_index, chunks, embeddings):
    from pinecone.core.openapi.shared.exceptions import UnauthorizedException

    logger.info(
        f"Starting to store vectors. Chunks: {len(chunks)}, Embeddings: {len(embeddings)}"
    )

    # Validate inputs
    if len(chunks) != len(embeddings):
        logger.error(
            f"Mismatch in chunks and embeddings length: {len(chunks)} vs {len(embeddings)}"
        )
        return

    # Prepare vectors for upsert
    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        try:
            # Ensure embedding is a list
            embedding_list = (
                list(embedding) if not isinstance(embedding, list) else embedding
            )

            # Log details about each vector
            # logger.info(
            #     f"Vector {i}: Chunk length {len(chunk)}, Embedding length {len(embedding_list)}"
            # )

            # Prepare vector for upsert
            vectors_to_upsert.append(
                (str(i), embedding_list, {"text": chunk})  # ID  # Vector  # Metadata
            )
        except Exception as e:
            logger.error(f"Error preparing vector {i}: {e}")

    # Perform batch upsert
    try:
        # Upsert in batches to handle large datasets
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            logger.info(f"Upserting batch {i//batch_size + 1}: {len(batch)} vectors")

            try:
                _index.upsert(batch)
                logger.info(f"Successfully upserted batch {i//batch_size + 1}")
            except UnauthorizedException as api_err:
                logger.error(
                    f"Pinecone API error in batch {i//batch_size + 1}: {api_err}"
                )
            except Exception as batch_err:
                logger.error(f"Error upserting batch {i//batch_size + 1}: {batch_err}")

        logger.info("Completed vector storage in Pinecone")

    except Exception as e:
        logger.error(f"Critical error storing vectors in Pinecone: {e}")
        raise
