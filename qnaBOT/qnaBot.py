import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Contextual Q&A Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Contextual Q&A Bot")
st.markdown("Upload a **CSV** or **Excel** file with **Question** and **Answer** columns to start chatting.")

# ==========================================
# 2. Caching Functions (Performance Optimization)
# ==========================================

@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

@st.cache_data
def generate_embeddings(_model, questions):
    return _model.encode(questions)

# ==========================================
# 3. Main Logic
# ==========================================

with st.sidebar:
    st.header("Configuration")
    
    # UPDATED: Added 'xlsx' and 'xls' to the accepted file types
    uploaded_file = st.file_uploader(
        "Upload your Knowledge Base", 
        type=['csv', 'xlsx', 'xls']
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=1.0, value=0.6, step=0.05
    )

if uploaded_file is not None:
    try:
        # UPDATED: Logic to handle CSV vs Excel
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            # For Excel files
            df = pd.read_excel(uploaded_file)

        # Validate Columns
        required_cols = ['Question', 'Answer']
        if not all(col in df.columns for col in required_cols):
            st.error(f"File must contain the following columns: {required_cols}")
        else:
            with st.expander("Preview Knowledge Base"):
                st.dataframe(df.head())

            with st.spinner("Initializing AI Brain..."):
                model = load_model()
                # Convert questions to string to handle potential formatting issues in Excel
                questions_list = df['Question'].astype(str).tolist()
                question_embeddings = generate_embeddings(model, questions_list)
            
            st.success("System Ready! Ask away below.")
            st.divider()

            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("What is your question?"):
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Search Logic
                user_vector = model.encode([prompt])
                similarity_scores = cosine_similarity(user_vector, question_embeddings)
                best_match_idx = np.argmax(similarity_scores)
                best_score = float(similarity_scores[0][best_match_idx])
                
                matched_question = df.iloc[best_match_idx]['Question']
                answer_text = df.iloc[best_match_idx]['Answer']

                if best_score >= confidence_threshold:
                    response = f"{answer_text} \n\n*(Confidence: {best_score:.2f})*"
                else:
                    response = (
                        f"I'm sorry, I couldn't find a confident answer based on the file. "
                        f"(Best match was: '{matched_question}' with {best_score:.2f} confidence)"
                    )

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("👋 Please upload a CSV or Excel file in the sidebar to begin.")