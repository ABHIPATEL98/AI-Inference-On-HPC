import os
import json
import base64
import pandas as pd
import streamlit as st
import shutil
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# --- NEW IMPORTS FOR HYBRID SEARCH ---
# BM25 is "Old School" search (like Ctrl+F but smarter). It looks for exact keyword matches.
from langchain_community.retrievers import BM25Retriever
# EnsembleRetriever mixes results from different searchers (Vector + Keyword).
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# These are for "Re-ranking" (The Judge). It double-checks the results to make sure they are actually relevant.
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- Constants ---
MODEL_PATH = "all-MiniLM-L6-v2" # The "Worker" AI for basic vectors
# The "Judge" AI. It is slower but much smarter at checking if text matches a question.
ENCODER_MODEL_PATH = "BAAI/bge-reranker-base" 

# Note: Ensure this directory exists or change to a temp folder
VECTORSTORE_PATH = "./vectorstore_demo/" 
if not os.path.exists(VECTORSTORE_PATH):
    os.makedirs(VECTORSTORE_PATH)

PROMPT_TEMPLATE = """
    Analyze the report and answer the questions as thoroughly as possible based on the content if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
"""

# --- Utility Functions ---

@st.cache_resource
def load_and_split_pdf(file_path, chunk_size=1000, chunk_overlap=20):
    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

@st.cache_resource
def create_embeddings_from_chunks(_chunks, model_path, store_path):
    """
    Creates TWO types of search indexes:
    1. Vector Store (FAISS) for meaning-based search.
    2. BM25 Retriever for keyword-based search.
    """
    # 1. Setup Vector Search (The "Vibe" Search)
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if you have a GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(_chunks, embedding_model)
    vectorstore.save_local(store_path)
    
    # 2. Setup Keyword Search (The "Exact Word" Search)
    # This helps catch specific names or technical terms that Vector search might miss.
    bm25_retriever = BM25Retriever.from_documents(_chunks)
    bm25_retriever.k = 5 # Ask BM25 for its top 5 candidates
    
    return vectorstore, bm25_retriever

def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def initialize_retriever(store_path, model_path, encoder_model_path, file_path):
    """
    Initializes the sophisticated Hybrid Search pipeline.
    Flow: User Query -> (Vector Search + Keyword Search) -> Mix Results -> Re-rank (Judge) -> Top Results
    """
    _chunks = load_and_split_pdf(file_path)
    
    # Step 1: Create both search engines
    vectorstore, bm25_retriever = create_embeddings_from_chunks(_chunks, model_path, store_path)
    
    # Configure the Vector Retriever
    faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    
    # Step 2: Create the Ensemble (The Mixer)
    # This asks BOTH engines for results and combines them into one list.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5] # Give equal importance to Keywords and Vectors
    )

    # Step 3: Re-ranking (The Judge)
    # The Ensemble might return 10 results (5 from each). Some might be irrelevant.
    # The CrossEncoder looks at the specific Question + Answer pair and scores them accurately.
    model = HuggingFaceCrossEncoder(model_name=encoder_model_path, model_kwargs={'device': 'cpu'})
    
    # Step 4: Compression
    # This filters the list down to only the Top 6 *highest quality* matches.
    compressor = CrossEncoderReranker(model=model, top_n=6) 
    
    # The final tool that runs this whole pipeline
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
    
    return compression_retriever

def setup_rag_chain(_retriever):
    prompt_template = PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_template(prompt_template)
    # Ensure Ollama is running locally with 'llama3' model pulled
    llm = ChatOllama(model="llama3", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_documents(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel({"context": _retriever, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

def display_pdf(file_path):
    """
    Display PDF in Streamlit using an iframe
    """
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit Application
def main_content():
    # --- SIDEBAR: Uploader ---
    with st.sidebar:
        st.title("Welcome")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Initialize variables
    file_path = None
    
    if uploaded_file is not None:
        # Save uploaded file consistently so we can display it
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Split the screen into two columns
    col1, col2 = st.columns([1, 1]) # 50% width each

    # --- LEFT COLUMN: PDF Display ---
    with col1:
        st.header("Document Viewer")
        if file_path:
            display_pdf(file_path)
        else:
            st.info("Upload a PDF in the sidebar to view it here.")

    # --- RIGHT COLUMN: Chat Logic ---
    with col2:
        st.header("Chat Interface")
        if uploaded_file is not None and file_path:
            # Initialize Retriever
            if "retriever" not in st.session_state:
                with st.spinner("Processing PDF with Hybrid Search..."):
                    # We now pass the extra Encoder path to the initializer
                    st.session_state.retriever = initialize_retriever(
                        VECTORSTORE_PATH, 
                        MODEL_PATH, 
                        ENCODER_MODEL_PATH, 
                        file_path
                    )
            retriever = st.session_state.retriever

            # Initialize Chain
            if "rag_chain_with_source" not in st.session_state:
                st.session_state.rag_chain_with_source = setup_rag_chain(retriever)
            rag_chain_with_source = st.session_state.rag_chain_with_source

            # Input for user question - Hitting Enter triggers the logic below
            user_question = st.text_input("Ask a question about the document:")

            if user_question:
                # Direct search on user query - removed query generation step
                st.write("### Answer")
                output = {}
                output_placeholder = st.empty()
                
                # Stream the response
                for chunk in rag_chain_with_source.stream(user_question):
                    for key, value in chunk.items():
                        if key not in output:
                            output[key] = value
                        else:
                            output[key] += value

                    if output.get('answer'):
                        output_placeholder.markdown(f"{output['answer']}")

                # Show relevant docs
                relevant_docs_list = retriever.get_relevant_documents(user_question) 
                formatted_docs = format_documents(relevant_docs_list)
                
                with st.expander("View Source Context"):
                    st.text(formatted_docs)
        
        else:
            st.info("Please upload a PDF to start chatting.")

def main():
    # Set page to wide mode to accommodate the side-by-side view
    st.set_page_config(layout="wide", page_title="FreeBot", page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>Hybrid RAG Based PDF Assistant</h1>", unsafe_allow_html=True)
    main_content()

if __name__ == "__main__":
    main()