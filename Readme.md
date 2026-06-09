# 📄 AI Document Assistant

An AI-powered Document Intelligence Platform built using Streamlit, Pinecone, Hugging Face, and Qwen. The application allows users to upload PDF documents, generate summaries, and ask questions using Retrieval-Augmented Generation (RAG).

## 🚀 Features

* Upload PDF documents
* Automatic text extraction using PyMuPDF
* Document summarization using Qwen LLM
* Text chunking for efficient retrieval
* Semantic embeddings using Sentence Transformers
* Vector storage and search using Pinecone
* Retrieval-Augmented Generation (RAG)
* Context-aware question answering
* Interactive Streamlit interface

## 🏗️ Architecture

```text
PDF Upload
     ↓
Text Extraction (PyMuPDF)
     ↓
Chunking
     ↓
Embeddings
     ↓
Pinecone Vector Database
     ↓
Semantic Retrieval
     ↓
Qwen LLM
     ↓
Answer Generation
```

## 🛠️ Tech Stack

* Python
* Streamlit
* Pinecone
* Hugging Face
* Qwen
* Sentence Transformers
* PyMuPDF
* LangChain Text Splitters

## 📂 Project Structure

```text
AI-Document-Assistant/
│
├── app.py
├── requirements.txt
├── .env
│
├── rag/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrival.py
│   └── vector_db.py
│
├── llm/
│   ├── summary.py
│   └── qa.py
│
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd AI-Document-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=researchai
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 📋 Usage

1. Upload a PDF document.
2. Generate a document summary.
3. Store document chunks in Pinecone.
4. Ask questions about the document.
5. Receive context-aware answers generated using RAG.

## 🎯 Future Improvements

* Multi-document support
* Citation-aware answers
* Source page references
* Document type detection
* Export summaries to PDF
* Chat history support
* Advanced analytics dashboard

## 👨‍💻 Author

Nikhil Kant

## 📜 License

This project is open source and available under the MIT License.
