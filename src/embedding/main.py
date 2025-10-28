import json
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from logger_conf import get_logger

load_dotenv()
log = get_logger(__name__)
PROCESSED_JSON_FOLDER = os.getenv("OUT_FOLDER")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
os.makedirs("vectordb/", exist_ok=True)
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
NOISE_ELEMENTS = ["Header", "Footer", "PageNumber", "Title"]


def main():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device="cpu"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_func
    )
    log.info("ChromaDB initialized successfully in embedding")
    json_files = [f for f in os.listdir(PROCESSED_JSON_FOLDER) if f.endswith(".json")]
    total_chunks_ingested = 0

    for json_file in json_files:
        file_path = os.path.join(PROCESSED_JSON_FOLDER, json_file)
        log.info(f"\nProcessing file: {json_file}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        documents = []
        metadatas = []
        ids = []

        file_chunks_count = 0

        for element in data.get("elements", []):
            element_type = element.get("category")
            if element_type not in NOISE_ELEMENTS:
                text = element.get("text_as_html") or element.get("text")
                if not text:
                    continue
                documents.append(text)
                metadatas.append(
                    {
                        "source_file": data.get("filename"),
                        "page_number": element.get("page_number"),
                        "element_type": element_type,
                    }
                )

                ids.append(f"{data.get('filename')}_el_{element.get('element_index')}")
                file_chunks_count += 1

        if documents:
            log.info(f"  Found {file_chunks_count} useful chunks. Adding to ChromaDB")
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            total_chunks_ingested += len(documents)
        else:
            log.info("  No useful chunks found in this file.")

    log.info(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()