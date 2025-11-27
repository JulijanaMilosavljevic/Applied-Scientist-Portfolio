# 🔍 RAG Search Assistant (FLAN-T5 + FAISS + Sentence Transformers)

This project implements a lightweight yet powerful **Retrieval-Augmented Generation (RAG)** system using:

- **FLAN-T5 Base** for answer generation  
- **Sentence Transformers (all-MiniLM-L6-v2)** for embeddings  
- **FAISS** for fast vector similarity search  
- **Gradio UI** for an interactive assistant demo  

The system takes a user question, retrieves the most relevant document chunks, and generates a grounded answer using FLAN-T5.

This project is optimized to run **entirely on CPU**, making it ideal for:
- student portfolios  
- junior applied scientist preparation  
- GitHub demonstration  
- interview discussion  
## ✨ Project Features

This RAG system includes the full end-to-end pipeline used in real-world retrieval applications:

### 🔹 1. Document Loading  
Automatically loads all `.txt` and `.md` files from the `data/documents/` directory.

### 🔹 2. Text Chunking  
Uses **RecursiveCharacterTextSplitter** to split documents into small, overlapping chunks for efficient retrieval.

### 🔹 3. Embedding Generation  
Creates dense semantic vectors using  
`sentence-transformers/all-MiniLM-L6-v2`.

### 🔹 4. FAISS Vector Store  
Builds a FAISS index for high-speed similarity search.

### 🔹 5. Retrieval Pipeline  
Given a user query, retrieves the **top-k** most relevant chunks.

### 🔹 6. FLAN-T5 Based Generation  
Combines retrieved context + question into a structured prompt and generates grounded answers.

### 🔹 7. Evaluation Tools  
- tabular overview of retrieved chunks  
- inspection of context relevance  
- visualization utilities

### 🔹 8. Interactive Gradio UI  
A clean Web UI where users can ask questions and see:
- the generated answer  
- retrieved chunks  
- how the RAG system works internally  

### 🔹 9. Fully CPU-Friendly  
Works smoothly on machines without GPU.
## 📂 Folder Structure

The project follows a clean and modular structure:

```bash
06-rag-search-assistant/
│
├── data/
│ └── documents/ # Raw text/markdown files used as knowledge base
│
├── models/ # (Ignored) Generated FLAN-T5 model artifacts
│
├── index/ # (Ignored) FAISS vector store files
│
├── static/ # Optional visualizations or images
│
├── rag_search_assistant.ipynb # Main notebook with the full RAG pipeline
│
└── README.md # Project documentation
```

### 📝 Notes:
- The **`data/documents/`** folder contains 4 real knowledge documents:
  - `ml_overview.txt`
  - `nlp_foundations.txt`
  - `embeddings_vector_stores.txt`
  - `rag_concept.txt`

- The folders **`models/`** and **`index/`** are intentionally excluded using `.gitignore`  
  because they are automatically generated and often large.
## 🛠️ Installation & Requirements

This project is fully CPU-friendly and runs on any standard machine.

### 📦 1. Clone the Repository

```bash
git clone https://github.com/JulijanaMilosavljevic/Applied-Scientist-Portfolio.git
cd Applied-Scientist-Portfolio/06-rag-search-assistant
```
### 🔧 2. Install Dependencies

You can install all required packages using:
```bash
pip install -r requirements.txt
```
### 🧰 3. Key Libraries Used

- transformers — FLAN-T5 generation

- sentence-transformers — embeddings

- faiss-cpu — fast vector search

- gradio — interactive UI

- langchain — chunking utilities

- pandas — retrieval inspection

- numpy — vector operations

#### 💻 Hardware Requirements

- CPU-only is fully supported

- 4GB RAM is enough

- No GPU required
## 🚀 Usage

The entire RAG workflow is implemented inside the  
**`rag_search_assistant.ipynb`** notebook.

To run the project end-to-end:

### ▶️ 1. Open the Notebook

```bash
jupyter notebook rag_search_assistant.ipynb
```
### ▶️ 2. Run All Cells

The notebook automatically performs:

1. Load documents from data/documents/

2. Split into chunks using RecursiveCharacterTextSplitter

3. Generate embeddings with Sentence Transformers

4. Index vectors inside a FAISS store

5. Retrieve top-k chunks for any user query

6. Build a structured prompt combining context + question

7. Generate grounded answer using FLAN-T5 Base

8. Display retrieved chunks in a clean table

9. Launch Gradio UI for interactive question answering

Everything runs fully on CPU and requires
no additional configuration.
