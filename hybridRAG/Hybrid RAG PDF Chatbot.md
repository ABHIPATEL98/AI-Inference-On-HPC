Hybrid RAG PDF Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain. This chatbot allows users to upload PDF documents and ask questions, providing accurate answers by combining keyword search, vector search, and AI re-ranking.

🌟 Key Features

Hybrid Search: Combines BM25 (keyword matching) and FAISS (semantic vector search) to find the best context.

Re-Ranking: Uses a Cross-Encoder (BAAI/bge-reranker-base) to "judge" and re-score search results for higher accuracy.

Local Privacy: Runs entirely on your machine using Ollama for the LLM (Llama 3).

Interactive UI: Built with Streamlit for easy PDF uploading and side-by-side document viewing.

Source Citations: Shows exactly which parts of the document were used to generate the answer.

🛠️ Prerequisites

Before running the app, ensure you have the following installed:

Python 3.8+

Ollama: Download from ollama.com.

Llama 3 Model: Run the following command in your terminal to pull the model:

ollama pull llama3


📦 Installation

Clone the repository (or save the python file):

git clone <your-repo-url>
cd <your-repo-folder>


Install Python Dependencies:
You need to install Streamlit, LangChain, and the specific search libraries.

pip install streamlit pandas langchain langchain-community langchain-core pymupdf faiss-cpu sentence-transformers rank_bm25


(Note: If you have a GPU, you can install faiss-gpu instead of faiss-cpu for faster performance).

🚀 Usage

Start the Ollama Server:
Ensure Ollama is running in the background. On Mac/Linux, it usually runs automatically. On Windows, open the Ollama application.

Run the Streamlit App:

streamlit run hybrid_app_commented.py


Interact with the App:

Open your browser (usually at http://localhost:8501).

Sidebar: Click "Browse files" and upload a PDF.

Chat: Type your question in the text input box.

View Sources: Expand the "View Source Context" dropdown to see the text segments the AI used.

🧠 How It Works

Ingestion: The uploaded PDF is split into 1000-character chunks.

Indexing:

Chunks are converted to vectors using all-MiniLM-L6-v2 and stored in FAISS.

Chunks are indexed for keywords using BM25.

Retrieval (Ensemble): When you ask a question, the app fetches top results from both indexes and combines them.

Re-Ranking: The BAAI/bge-reranker-base model examines the combined results and selects only the top 6 most relevant chunks.

Generation: These top chunks are sent to Llama 3 to generate the final answer.

⚙️ Configuration

You can modify the constants at the top of hybrid_app_commented.py to customize the app:

MODEL_PATH: Change the embedding model (default: all-MiniLM-L6-v2).

ENCODER_MODEL_PATH: Change the re-ranking model (default: BAAI/bge-reranker-base).

chunk_size: Adjust how large the text splits are.

⚠️ Note on Performance

The first time you run a query, the application needs to download the embedding and re-ranking models from HuggingFace. This might take a minute depending on your internet connection. Subsequent runs will be faster.