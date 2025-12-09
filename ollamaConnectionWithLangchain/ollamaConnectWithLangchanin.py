import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# --- 1. WEB PAGE SETUP ---
# This sets the big heading text at the top of your web page.
st.title("ollama with langchain bot (Q&A Only)")

# --- 2. SETTING UP THE AI MODEL ---
# @st.cache_resource is a special Streamlit command.
# It tells the app: "Load this model ONLY ONCE and remember it."
# If we didn't use this, the app would reload the entire 8GB model every time you clicked a button!
@st.cache_resource
def load_model():
    print("Initializing Ollama model...") # This prints to your computer's black terminal window
    
    # We configure the AI model here
    model = ChatOllama(
        model="llama3.1:8b",                # This is the name of the brain we are using
        base_url="http://localhost:11434/", # This is the address where Ollama is running on your PC
        verbose=True,                       # distinct logs for debugging
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
        temperature=0                       # 0 = Factual/Precise. 1 = Creative/Random.
    )
    return model

# calling the function to actually get the model ready
model = load_model()

# --- 3. CREATING THE USER INTERFACE ---
# Create a text box where the user can type their question.
user_query = st.text_input("Ask a question:")

# Create a button. The code inside this 'if' block ONLY runs when:
# 1. The user clicks "Get Answer" AND
# 2. The 'user_query' box is not empty
if st.button("Get Answer") and user_query:
    
    # --- 4. PREPARING THE INSTRUCTION (PROMPT) ---
    # Since this is NOT conversational, we don't send old messages.
    # We just send the specific instructions and the ONE question the user just asked.
    template = """
    You are a helpful AI Assistant. Explain things in short and brief.
    
    User Question: {question}
    """
    
    # This converts our text string into a format the AI library (LangChain) understands
    prompt = ChatPromptTemplate.from_template(template)
    
    # --- 5. RUNNING THE AI ---
    # st.spinner shows a little "loading" circle while the AI thinks
    with st.spinner("Generating answer..."):
        
        # This is the "Chain" (The Pipeline).
        # It reads like a pipe: Prompt -> goes into -> Model -> goes into -> Text Parser.
        # The '|' symbol literally means "pass the output of the left side to the right side".
        chain = prompt | model | StrOutputParser()
        
        st.write("### Answer:")
        
        # --- 6. DISPLAYING THE RESULT ---
        # chain.stream() asks the AI to generate the answer piece-by-piece.
        # st.write_stream() grabs those pieces and puts them on screen immediately (Typewriter effect).
        # We pass the user's question into the {question} placeholder we made in step 4.
        st.write_stream(chain.stream({"question": user_query}))