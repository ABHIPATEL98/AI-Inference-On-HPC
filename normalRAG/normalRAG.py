import os
import json
import base64
import pandas as pd
import streamlit as st
import shutil

# --- LangChain Imports ---
# These are the tools needed to process text and talk to AI
from langchain.embeddings import HuggingFaceEmbeddings  # Converts text into numbers (vectors)
from langchain.document_loaders import PyMuPDFLoader    # Reads PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter # Chops text into smaller pieces
from langchain.vectorstores import FAISS                # A database optimized for searching similar text
from langchain.chat_models import ChatOllama            # The connection to the Ollama AI server
from langchain.prompts import ChatPromptTemplate        # Helps format instructions for the AI
from langchain.callbacks.manager import CallbackManager # Manages events (like streaming text)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # Allows text to appear as it's being generated
from langchain_core.runnables import RunnableParallel, RunnablePassthrough # Helps chain steps together
from langchain_core.output_parsers import StrOutputParser # Cleans up the AI's output into a string
from datetime import datetime

# --- Constants ---
# The specific model used to turn text into numbers. This one is small and fast.
MODEL_PATH = "all-MiniLM-L6-v2"

# Where we will save the "number database" (Vector Store).
VECTORSTORE_PATH = "./vectorstore_demo/" 

# Create the folder if it doesn't exist
if not os.path.exists(VECTORSTORE_PATH):
    os.makedirs(VECTORSTORE_PATH)

# The instructions we give the AI. We call this the "System Prompt".
PROMPT_TEMPLATE = """
    Analyze the report and answer the questions as thoroughly as possible based on the content if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
"""

# --- Utility Functions ---

# @st.cache_resource tells Streamlit: "Run this once and remember the result."
# This prevents reloading the PDF every time you click a button.
@st.cache_resource
def load_and_split_pdf(file_path, chunk_size=1000, chunk_overlap=20):
    """
    1. Loads the PDF.
    2. Cuts it into small pieces (chunks) so the AI can digest it.
    """
    # Load the PDF file
    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()
    
    # Setup the splitter. 
    # chunk_size=1000: Each piece of text is 1000 characters long.
    # chunk_overlap=20: The end of one piece overlaps with the start of the next (helps keep context).
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Return the split documents
    return text_splitter.split_documents(documents)

@st.cache_resource
def create_embeddings_from_chunks(_chunks, model_path, store_path):
    """
    1. Takes the text chunks.
    2. Uses a model to turn them into lists of numbers (Embeddings).
    3. Saves them into a FAISS database (Vector Store).
    """
    # Initialize the model that converts text -> numbers
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'}, # Use 'cpu' so it works on any computer. Use 'cuda' if you have an NVIDIA GPU.
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create the database (Vector Store) using the chunks and the model
    vectorstore = FAISS.from_documents(_chunks, embedding_model)
    
    # Save it to the hard drive so we don't have to rebuild it instantly
    vectorstore.save_local(store_path)
    return vectorstore

def format_documents(docs):
    """
    Helper function to combine multiple text chunks into one big string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def initialize_retriever(store_path, model_path, file_path):
    """
    The 'Retriever' is the tool that searches the database.
    This function coordinates the whole loading process.
    """
    # Step 1: Load and Split PDF
    _chunks = load_and_split_pdf(file_path)
    
    # Step 2: Create the Database (Vector Store)
    vectorstore = create_embeddings_from_chunks(_chunks, model_path, store_path) 
    
    # Step 3: Turn the database into a "Retriever"
    # k=5 means "Find the top 5 most similar pieces of text to the user's question"
    faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    
    return faiss_retriever

def setup_rag_chain(_retriever):
    """
    This builds the 'Assembly Line' (Chain) for the AI.
    RAG = Retrieval Augmented Generation.
    """
    prompt_template = PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Initialize the LLM (Large Language Model)
    # You MUST have Ollama installed and run `ollama pull llama3` in your terminal for this to work.
    llm = ChatOllama(
        model="llama3", 
        verbose=True, 
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
        temperature=0 # 0 means "be factual", 1 means "be creative"
    )

    # This defines the steps the AI takes to get the answer:
    # 1. Take the formatted documents (context)
    # 2. Add the Prompt instructions
    # 3. Send to the LLM
    # 4. Clean up the output (StrOutputParser)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_documents(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # This combines the Retriever with the generation chain
    return RunnableParallel({"context": _retriever, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

def display_pdf(file_path):
    """
    Shows the PDF inside the web browser using HTML code (iframe).
    """
    with open(file_path, "rb") as f:
        # Convert the PDF file to a base64 string so HTML can read it
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # HTML code to embed the PDF
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    
    # Render the HTML
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- Streamlit Application (The Main UI) ---
def main_content():
    # --- SIDEBAR: Upload File ---
    with st.sidebar:
        st.title("Welcome")
        # Widget to let user drop a file
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    file_path = None
    
    if uploaded_file is not None:
        # We need to save the file to the disk temporarily to process it
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Create two columns layout (Left for PDF, Right for Chat)
    col1, col2 = st.columns([1, 1]) 

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
        
        # Only start the chat logic if a file is uploaded
        if uploaded_file is not None and file_path:
            
            # --- Initialize the "Brain" ---
            # session_state is how Streamlit remembers things between clicks.
            if "retriever" not in st.session_state:
                with st.spinner("Processing PDF..."):
                    # This runs the heavy loading/embedding functions
                    st.session_state.retriever = initialize_retriever(VECTORSTORE_PATH, MODEL_PATH, file_path)
            
            retriever = st.session_state.retriever

            # --- Initialize the "Chain" ---
            if "rag_chain_with_source" not in st.session_state:
                st.session_state.rag_chain_with_source = setup_rag_chain(retriever)
            
            rag_chain_with_source = st.session_state.rag_chain_with_source

            # --- User Input ---
            user_question = st.text_input("Ask a question about the document:")

            if user_question:
                st.write("### Answer")
                output = {}
                output_placeholder = st.empty() # Creates a space to fill text into later
                
                # --- Stream the Response ---
                # We loop through the response as it comes in (streaming)
                # so the user sees the text typing out.
                for chunk in rag_chain_with_source.stream(user_question):
                    for key, value in chunk.items():
                        if key not in output:
                            output[key] = value
                        else:
                            output[key] += value

                    # Update the placeholder with the new text
                    if output.get('answer'):
                        output_placeholder.markdown(f"{output['answer']}")

                # --- Show Sources ---
                # Find the specific parts of the PDF used to answer the question
                relevant_docs_list = retriever.get_relevant_documents(user_question) 
                formatted_docs = format_documents(relevant_docs_list)
                
                # Create a collapsible box to show the raw text sources
                with st.expander("View Source Context"):
                    st.text(formatted_docs)
        
        else:
            st.info("Please upload a PDF to start chatting.")

def main():
    # Configure the browser tab title and layout
    st.set_page_config(layout="wide", page_title="FreeBot", page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>FreeBot PDF Assistant</h1>", unsafe_allow_html=True)
    main_content()

if __name__ == "__main__":
    main()