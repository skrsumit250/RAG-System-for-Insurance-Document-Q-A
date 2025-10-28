import json
import os

from dotenv import load_dotenv
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from logger_conf import get_logger

log = get_logger(__name__)
load_dotenv()
PDF_FOLDER_PATH = os.getenv("PDF_FOLDER_PATH")
OUT_FOLDER = os.getenv("OUT_FOLDER")
os.makedirs(OUT_FOLDER, exist_ok=True)

all_pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.endswith(".pdf")]

print(f"Found {len(all_pdf_files)} PDF files to process.")


def _meta_get(meta, key):
    if meta is None:
        return None
    if hasattr(meta, key):
        return getattr(meta, key)
    if isinstance(meta, dict):
        return meta.get(key)
    return None


for fname in all_pdf_files:

    pdf_path = os.path.join(PDF_FOLDER_PATH, fname)
    # print(pdf_path)
    log.info(f"\nProcessing file: {fname}")

    try:
        partition = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            infer_table_structure=True,
            extract_images_in_pdf=False,
        )
        elements = chunk_by_title(
            elements=partition, combine_text_under_n_chars=2000, max_characters=4000
        )
    except Exception as e:
        log.error(f" Could not process {fname}: {e}")
        continue

    log.info(f"  Extracted {len(elements)} elements.")

    out = {"filename": fname, "num_elements": len(elements), "elements": []}

    for i, el in enumerate(elements):
        metadata = getattr(el, "metadata", None)
        out["elements"].append(
            {
                "element_index": i,
                "category": getattr(el, "category", None),
                "page_number": _meta_get(metadata, "page_number"),
                "text": el.text,
                "text_as_html": _meta_get(metadata, "text_as_html"),
            }
        )

    out_path = os.path.join(OUT_FOLDER, os.path.splitext(fname)[0] + ".json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log.info(f"  Successfully saved to {out_path}")

log.info("\nAll files processed.")