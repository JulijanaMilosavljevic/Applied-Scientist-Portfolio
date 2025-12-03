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
## 🧩 RAG Pipeline Breakdown

This project implements a full Retrieval-Augmented Generation (RAG) pipeline.  
Below is a detailed overview of each component and how they interact.

---

### 🔹 1. Document Loading

All `.txt` and `.md` files inside `data/documents/` are automatically loaded into memory.

This forms the **knowledge base** for the assistant.

---

### 🔹 2. Text Chunking (RecursiveCharacterTextSplitter)

Documents are split into overlapping chunks:

- **chunk_size = 350 characters**
- **chunk_overlap = 50 characters**

Why chunking?

- improves retrieval precision  
- provides better semantic context  
- avoids truncation issues  
- reduces noise in vector search  

Chunking is essential for stable RAG performance.

---

### 🔹 3. Embeddings (Sentence Transformers)

Each chunk is converted into a dense semantic vector using:

```bash
sentence-transformers/all-MiniLM-L6-v2
```

Benefits:

- lightweight  
- fast on CPU  
- surprisingly strong semantic performance  
- ideal for small-scale RAG systems  

---

### 🔹 4. Vector Store (FAISS)

All embeddings are stored in a **FAISS index**.

FAISS provides:

- extremely fast similarity search  
- optimized CPU performance  
- scalable vector databases  

---

### 🔹 5. Retrieval (Top-k Similarity Search)

Given a user query:

1. Embed the query  
2. Search FAISS  
3. Return **top-k most relevant chunks**  
4. Provide them to the generator model  

This ensures the answer is **grounded in retrieved evidence**.

---

### 🔹 6. Prompt Construction

We build a clean, instruction-style prompt:

```bash
You are an AI assistant. Use ONLY the provided context..
```

The prompt contains:

- Retrieved chunks  
- The user question  
- Safety rules (no guessing)  

---

### 🔹 7. Generation (FLAN-T5 Base)

FLAN-T5 is an instruction-fine-tuned model:

- excellent for Q&A  
- compact but strong  
- CPU-friendly  
- fast inference  

The model generates a final answer **based strictly on the retrieved context**.

---

### 🔹 8. RAG Answer Output

Final output consists of:

- 🤖 **Generated Answer**  
- 📄 **Retrieved Chunks**  
- 🔍 **Grounding Evidence**

This ensures transparency and prevents hallucination.

---
## 🧪 Examples

Below are sample inputs and outputs demonstrating how the RAG system retrieves relevant context and generates grounded answers using FLAN-T5.

---

### **🔸 Example 1 — “What is deep learning?”**

**User Query:**  
> *What is deep learning?*

**Retrieved Chunks:**  
- Explains that deep learning is a subfield of machine learning  
- Mentions neural networks and representation learning  
- Provides definitions from the ML overview document  

**Generated Answer:**  
> Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns and representations in data.

---

### **🔸 Example 2 — “Explain embeddings in ML.”**

**User Query:**  
> *Explain embeddings in machine learning.*

**Retrieved Chunks:**  
- Describes vector representations  
- Explains semantic similarity  
- Details high-dimensional encoding  

**Generated Answer:**  
> Embeddings are vector representations of data that capture semantic meaning. They allow models to measure similarity by comparing distances in vector space.

---

### **🔸 Example 3 — “What is the purpose of a vector database?”**

**User Query:**  
> *What is the purpose of a vector database?*

**Retrieved Chunks:**  
- Mentions FAISS, vector search, ANN, and dense embeddings  
- Explains indexing and retrieval  

**Generated Answer:**  
> A vector database stores embedding vectors and enables fast similarity search so the system can retrieve the most relevant information for a query.

---

### **🔸 Example 4 — “How does RAG work?”**

**User Query:**  
> *How does Retrieval-Augmented Generation (RAG) work?*

**Retrieved Chunks:**  
- Overview of retrieval, chunking, embeddings, and generation  
- Explanation from the `rag_concept.txt` document  

**Generated Answer:**  
> RAG combines document retrieval with generative models. It retrieves relevant chunks, inserts them into a prompt, and the generator model produces an answer grounded in this context.
## 💬 Gradio UI — Interactive RAG Assistant

The project includes a full **Gradio interface** that allows users to interact with the RAG system in real time.

With this UI, you can:

- type any question into the input box  
- retrieve the most relevant document chunks  
- generate a grounded answer using FLAN-T5  
- inspect the retrieved context for transparency  

---

### 🔧 **How to Launch the UI**

Inside the notebook, simply run:

```python
demo.launch()
```
Or, to display the UI directly inside Jupyter Notebook:
```python
demo.launch(inline=True)
```
### 🖥️ UI Preview

- When the interface launches, it shows:

- A text field for entering your query

- A Generate Answer button

- A text box with the RAG Answer

- A text box showing all Retrieved Chunks

- This allows full transparency of the retrieval process and demonstrates how RAG systems ground responses in real documents.

### 🎯 Why This Matters

- Interactive UIs are a strong component of applied projects because they show:

- practical usability

- real-time inference

- retrieval transparency

- model grounding

- user-facing AI development skills
## 📊 Evaluation Tools

To better understand how well the system retrieves and uses information, this project includes several evaluation utilities.

These tools help analyze:

- whether the retrieved context is relevant  
- if the answer is grounded in the documents  
- how FAISS search behaves for different queries  

---

### 🔹 1. Retrieved Chunks Table

A clean `pandas` DataFrame displays all the chunks retrieved for a query.

Example:

| Retrieved Chunk |
|-----------------|
| "Deep learning is a subset of machine learning..." |
| "Neural networks allow models to learn patterns..." |
| "Representation learning is central to deep learning..." |

This table makes it easy to inspect the context that FLAN-T5 uses.

---

### 🔹 2. Retrieval Visualization

The project includes an optional function that generates a visual summary image:

- the query  
- the selected chunks  
- the beginning of each chunk  

This is useful for presentations and documentation.

---

### 🔹 3. Why Evaluation Matters

Evaluation provides insights into:

- retrieval quality  
- grounding accuracy  
- chunking effectiveness  
- embedding performance  
- FAISS search behavior
## 📝 Project Notes

### 🔸 1. Models and Index Files Are Not Included in GitHub

To keep the repository clean and lightweight, the following folders are **excluded** using `.gitignore`:

- `models/` — FLAN-T5 tokenizer & model artifacts  
- `index/` — FAISS vector store  
- temporary cache files  
- large binary artifacts  

These files are generated automatically when running the notebook and **should not** be version-controlled.

This ensures:

- fast repository cloning  
- no large unnecessary files  
- clean version history  
- reproducible pipeline setup  

---

### 🔸 2. Documents Folder

Only raw `.txt` and `.md` documents are stored in:

```bash
data/documents/
```

This makes the retrieval dataset transparent, editable, and easy to expand.

---

### 🔸 3. CPU-Optimized Pipeline

Every component in this project is intentionally chosen to run efficiently on CPU:

- MiniLM embeddings  
- FAISS-CPU  
- FLAN-T5 Base  
- lightweight chunking  
- optimized prompt templates  

This makes the solution ideal for students, demo applications, or interview projects.

---

### 🔸 4. Reproducible Notebooks

All steps in the notebook are self-contained:

- loading  
- chunking  
- embedding  
- indexing  
- retrieval  
- generation  
- evaluation  
- UI launch  

Anyone can run the pipeline end-to-end without additional setup.
## 🎓 What You Learn From This Project

This project demonstrates several key skills that are essential for modern AI/ML roles, including **Applied Scientist**, **ML Engineer**, and **AI Research Engineering** positions.

---

### 🔹 1. Retrieval-Augmented Generation (RAG)
Understanding how to combine retrieval systems with generative models is a must-have skill in the current LLM ecosystem.

This project shows that you understand:

- how retrieval improves grounding  
- why LLMs hallucinate  
- how to build context windows  
- how to generate answers based on evidence  

---

### 🔹 2. Semantic Embeddings
You gain practical experience with:

- sentence-transformers  
- vector representations  
- similarity search  
- embedding-based reasoning  

---

### 🔹 3. FAISS Vector Databases
You demonstrate that you can:

- build a vector index  
- perform top-k similarity search  
- optimize retrieval for speed and accuracy  

These are real-world skills used across industry.

---

### 🔹 4. Prompt Engineering for RAG
You learn how to structure prompts that:

- reduce hallucinations  
- clearly define task rules  
- insert retrieved context effectively  

---

### 🔹 5. Document Processing & Chunking
You show knowledge of:

- text preprocessing  
- overlapping windows  
- chunk optimization strategies  
- handling different document formats  

---

### 🔹 6. Lightweight Model Deployment
By integrating **FLAN-T5 Base**, you demonstrate that you can:

- run LLMs efficiently on CPU  
- avoid unnecessary dependencies  
- design portable AI systems  

---

### 🔹 7. Building Interactive AI Applications
With the Gradio UI, you gain experience in:

- turning ML pipelines into usable tools  
- creating clean interfaces  
- integrating retrieval + generation + UI  

---

### 🔹 8. End-to-End ML System Thinking
You show that you can design and implement a complete system, including:

- data ingestion  
- chunking  
- vectorization  
- indexing  
- retrieval  
- answer generation  
- evaluation  
- deployment
## 🚀 Future Improvements & Extensions

This project is intentionally lightweight and CPU-friendly, but it can be extended in several powerful ways to approach production-grade RAG systems.

---

### 🔹 1. Add Support for PDFs and Web Pages
Currently, the system processes `.txt` and `.md` files.  
Future upgrades may include:

- PDF parsing  
- HTML extraction  
- Web scraping pipelines  
- Multi-format document ingestion  

---

### 🔹 2. Improve Chunking Strategy
Enhancements may include:

- semantic chunking  
- dynamic chunk sizes  
- title-aware chunk splitting  
- hierarchical retrieval methods  

This improves both recall and precision.

---

### 🔹 3. Use Larger Embedding Models
For stronger retrieval quality:

- `sentence-transformers/all-mpnet-base-v2`  
- `bge-large-en`  
- `gte-large`  

These offer richer semantic representations (though slower on CPU).

---

### 🔹 4. Upgrade Generator Model
Possible alternatives to FLAN-T5:

- `flan-t5-large`  
- `mistral-7b-instruct` (if GPU available)  
- `llama-3-instruct`  
- quantized GGUF models  

This can significantly boost answer quality.

---

### 🔹 5. Add Re-ranking Step
Before sending context to the generator:

- use cross-encoder re-ranking  
- reorder retrieved chunks by semantic relevance  

This leads to cleaner and more accurate answers.

---

### 🔹 6. Memory & Conversation Support
Extend the assistant to:

- remember previous user queries  
- maintain running context  
- store conversation history  

This enables multi-turn assistant behavior.

---

### 🔹 7. Deploy the System
Potential deployment options:

- **Hugging Face Spaces** (simple, free)  
- **Render.com**  
- **Azure App Service**  
- **Docker container**  
- **FastAPI backend + Gradio frontend**  

This transforms the project into a fully accessible web application.

---

### 🔹 8. Add Retrieval Evaluation Metrics
Such as:

- Recall@k  
- MRR (Mean Reciprocal Rank)  
- NDCG  
- Chunk overlap analysis  

This brings the project closer to applied research workflows.

---

These improvements are optional, but they demonstrate a clear roadmap for growing this into a more advanced RAG system.
## ✅ Conclusion

This project demonstrates a complete, end-to-end **Retrieval-Augmented Generation (RAG)** system built entirely with CPU-friendly components.

It integrates:

- document processing  
- semantic embeddings  
- FAISS vector search  
- prompt engineering  
- LLM-based answer generation  
- an interactive Gradio interface  
- transparency through evaluation tools  

The result is a practical and interpretable assistant capable of answering questions based on a custom knowledge base.

This project is an excellent demonstration of skills for roles such as:

- Applied Scientist  
- ML Engineer  
- AI Research Engineer  
- NLP Engineer  

It shows not only the ability to train and run models, but also to design full AI systems, evaluate them, and present them through a usable interface.

---

### ⭐ Thank you for reviewing the project!

If you find the project helpful or inspiring, feel free to explore the rest of the portfolio.



