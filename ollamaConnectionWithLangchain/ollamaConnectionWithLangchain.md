Simple Ollama + LangChain Q&A Bot

A lightweight Streamlit application that interfaces with a local Ollama instance to provide a simple Question & Answer interface. This bot uses LangChain to manage the interaction and streams responses in real-time.

🌟 Key Features

Zero-Cost AI: Runs entirely on your local machine using the llama3.1:8b model via Ollama.

Instant Streaming: Uses Streamlit's streaming capabilities to type out answers as they are generated.

Simple Architecture: A minimal example of connecting a UI (Streamlit) to a Local LLM (Ollama) using LangChain.

Q&A Focused: Designed for single-turn questions (does not store chat history).

🛠️ Prerequisites

Before running the app, ensure you have the following installed:

Python 3.8+

Ollama: Download and install from ollama.com.

Llama 3.1 Model: Run the following command in your terminal to pull the specific model used in the code:

ollama pull llama3.1:8b


(Note: If you want to use a different model, you must update the model="llama3.1:8b" line in the Python script).

📦 Installation

Save the Code:
Save the provided Python code into a file named app.py.

Install Python Dependencies:
You need to install Streamlit and the LangChain integration for Ollama.

pip install streamlit langchain-ollama langchain-core


🚀 Usage

Start the Ollama Server:
Ensure Ollama is running in the background.

Mac/Linux: Usually runs automatically. Check with ollama serve.

Windows: Open the Ollama application from the taskbar.

Run the Streamlit App:
Navigate to the folder containing app.py and run:

streamlit run app.py


Interact with the App:

The app will open in your browser (usually http://localhost:8501).

Type your question in the text box (e.g., "Why is the sky blue?").

Click "Get Answer".

Watch the AI stream the response in real-time.

🧠 Code Overview

st.cache_resource: Ensures the heavy AI model is loaded only once when the app starts, not every time you interact with it.

ChatOllama: The LangChain connector that talks to your local Ollama server at http://localhost:11434/.

Streaming: The StreamlitCallbackHandler and write_stream functions allow the text to appear token-by-token, making the app feel faster and more responsive.

⚠️ Troubleshooting

Connection Error: If you see an error about connecting to localhost, make sure the Ollama app is actually running.

Model Not Found: If it says the model is missing, run ollama pull llama3.1:8b in your terminal.