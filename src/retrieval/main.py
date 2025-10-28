import os
import re

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder

from logger_conf import get_logger

log = get_logger(__name__)
load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
N_RESULTS_TO_RETRIEVE = 20
GROQ_MODEL = os.getenv("GROQ_MODEL")
groq_api_key = os.getenv("GROQ_API_KEY")


try:
    llm_client = Groq(api_key=groq_api_key)
    log.info("Groq client initialized.")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    log.info("Re-ranker initialized successfully")
except Exception as e:
    log.error(f"Error initializing Groq client: {e}")
    llm_client = None

log.info(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device="cpu"
    )
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=embedding_func
    )
    log.info(f"Successfully connected to db. It has {collection.count()} documents.")
except Exception as e:
    log.error(f"Could not connect to db: {e}")
    collection = None


def re_rank(retrieved_docs, metadatas, query_text):
    log.info("Starting re-rnaking")
    pairs = [(query_text, doc) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    # To combine ranks and scores
    ranked = sorted(
        zip(retrieved_docs, metadatas, scores), key=lambda x: x[2], reverse=True
    )
    top_docs = [doc for doc, meta, score in ranked[:3]]
    top_meta = [meta for doc, meta, score in ranked[:3]]

    return top_docs, top_meta


def retrieve_context(query_text):
    if collection is None:
        log.error("Database not found")
        return "Error: Database collection not found.", None

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=N_RESULTS_TO_RETRIEVE,
            include=["documents", "metadatas"],
        )
        # print(results)
        retrieved_docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        top_docs, top_meta = re_rank(retrieved_docs, metadatas, query_text)
        # return results["documents"][0], results["metadatas"][0]
        return top_docs, top_meta

    except Exception as e:
        log.error(f"Error during retrieval: {e}")
        return [], []


def generate_answer(query_text, context_chunks, context_metadata):
    if llm_client is None:
        return "Error: LLM client not initialized. Check API key."
    system_prompt = """
    You are an expert insurance assistant. Your task is to answer the user's
    query based *ONLY* on the provided context passages.

    Instructions:
    1.  Read the CONTEXT passages carefully.
    2.  Answer the user's QUERY using the information found in the CONTEXT.
    3.  **Cite your sources.** After each piece of information, add a citation
        in the format [source_file, page_number].
    5.  Use knowledge mentioned in context only.
    """

    context_str = ""
    for i in range(len(context_chunks)):
        doc = context_chunks[i]
        meta = context_metadata[i]
        context_str += f"""

CONTEXT PASSAGE {i+1}
Source File: {meta.get('source_file', 'N/A')}
Page Number: {meta.get('page_number', 'N/A')}

{doc}
---
"""
    user_prompt = f"""
    CONTEXT:
    {context_str}
    ---
    QUERY:
    {query_text}
    """
    try:
        # print("System prompt:", system_prompt)
        # print("\n\n\n\nUSer prompt", user_prompt)
        response = llm_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during Groq generation: {e}")
        return "Error: Failed to generate answer from Groq."


def ask_rag_pipeline(query_text):

    if collection is None:
        return "RAG system not initialized."
    log.info(f"Processing query. length is {len(query_text)}")
    chunks, metadata = retrieve_context(query_text)
    # Removes HTML tags
    answer = generate_answer(query_text, chunks, metadata)
    clean_answer = re.sub(r"<[^>]+>", "", answer)
    return clean_answer


if __name__ == "__main__":

    log.info("\nTesting RAG Core Pipeline")
    test_query = "For hip resurfacing, what are the symptoms?"

    final_answer = ask_rag_pipeline(test_query)

    log.info("\nFINAL RAG ANSWER")
    # print(final_answer)