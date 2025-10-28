# RAG System for Insurance Document Q&A

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer user questions based on a set of unstructured PDF insurance documents. The system provides answers and insights related to product manuals or policies, using a pipeline that includes chunking, embedding, retrieval with re-ranking, and LLM-based answer generation.

---

## 1. Project Overview and Goals

The primary goal is to create an automated Q&A system for product-related information contained in unstructured PDF manuals.
The RAG pipeline aims to:

1. **Process** unstructured PDF documents.
2. **Chunk** and **Embed** the document content.
3. **Retrieve** the most relevant context for a user query.
4. **Generate** a final, context-based answer.
5. **Evaluate** the solution using relevant metrics like ROUGE scores.

---

## 2. RAG Pipeline Components

The system is structured into several Python scripts corresponding to the steps of the RAG pipeline.

### 1. Chunking

This script handles the initial document ingestion and chunking process.

- **Input:** PDF documents.
- **Tool:** Uses `unstructured.partition_pdf` for text and table extraction.
- **Method:** Employs **title-based chunking** (`chunk_by_title`).
  - Uses `max_characters=4000` to control chunk.
- **Output:** Chunks and associated **metadata** (filename, page number, element type) are saved as individual JSON files.

### 2. Embedding

This script converts the document chunks into vectors and stores them in a vector database.

- **Vector Store:** **ChromaDB** is used for persistent storage..
- **Embedding Model:** A suitable **Sentence Transformer** model is employed.
- **Metadata Storage:** Stores metadata such as `source_file`, `page_number`, and `element_type` with vectors, excluding noisy elements like "Header" and "Footer".

### 3. Retrieval and Generation

This script implements the core Q&A logic integrating retrieval, re-ranking, and LLM generation.

- **Retrieval:** Queries ChromaDB for initial context using semantic search.
  - Retrieves `$N_RESULTS_TO_RETRIEVE = 20` results initially.
- **Re-ranking:** Uses a **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank documents for better relevance.
  - Selects top 3 re-ranked documents for final context.
- **Generation:** Uses the **Groq API** to generate the final answer.
  - The LLM is prompted as an expert insurance assistant and to answer **only** based on provided context.
  - **Citation:** Answers include citations in the format `[source_file, page_number]` to provide grounded responses.

### 4. Evaluation

This script evaluates the RAG system's performance through quantitative metrics.

- **Metrics:** Calculates **ROUGE-1, ROUGE-2, and ROUGE-L** F-measures against ground truth answers.
- **Test Data:** Loads evaluation data from `tests/questions.json`.
- **Output:** Prints average ROUGE scores and saves detailed results to `tests/rag_rouge_results.json`.

### 5. Chatbot Interface

A user-facing application built with Streamlit provides an interactive chat experience.

- **Tool:** Streamlit
- **Functionality:** Allows users to ask questions about insurance documents, displays AI-generated responses, and maintains chat history.

---

## 3. Getting Started

### Prerequisites

- Python 3.8+
- Environment variables set up in a `.env` file (e.g., `GROQ_API_KEY`, `PDF_FOLDER_PATH`, `EMBEDDING_MODEL`, etc.)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://git.tigeranalytics.com/trainings/ai_associate-program/springboard-batch/hackathon/team-4.git
    cd team-4
    ```
2. **Install dependencies:**
    ```bash
    conda env create -f env.yml
    ```
3. **Prepare Data:** Place your PDF insurance documents into the folder specified by `$PDF_FOLDER_PATH`.

---



