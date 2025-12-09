AI Contextual Q&A Bot

A smart, data-driven chatbot built with Streamlit that answers questions based strictly on a knowledge base you provide (CSV or Excel).

Unlike generative AI that makes up answers, this bot uses Semantic Search (via sentence-transformers) to find the most similar question in your uploaded file and returns the pre-written answer. This ensures 100% accuracy relative to your data.

🌟 Key Features

Custom Knowledge Base: Upload your own .csv, .xlsx, or .xls files.

Semantic Understanding: Uses the all-MiniLM-L6-v2 model to understand the meaning of questions, not just keyword matching. (e.g., "How much is it?" matches "What is the price?").

Confidence Control: An adjustable slider allows you to decide how strict the matching should be.

Instant Feedback: Displays the confidence score for every answer provided.

🛠️ Prerequisites

Python 3.8+

No API Keys Required: This runs entirely locally using open-source models.

📦 Installation

Save the Code:
Save your Python script as app.py.

Install Dependencies:
You need Streamlit, data processing libraries, and the AI model tools.

pip install streamlit pandas numpy sentence-transformers scikit-learn openpyxl


(Note: openpyxl is required for reading Excel files).

🚀 Usage

Run the App:

streamlit run app.py


Prepare Your Data:
Create a CSV or Excel file with exactly two columns:

Question: The potential questions users might ask.

Answer: The exact answer you want the bot to give.

Example data.csv:
| Question | Answer |
| :--- | :--- |
| What are your hours? | We are open 9am to 5pm, Mon-Fri. |
| Do you offer refunds? | Yes, refunds are available within 30 days. |
| Where is the office? | We are located at 123 Tech Lane. |

Interact:

Upload your file in the sidebar.

Wait a moment for the AI to "read" (embed) the questions.

Start chatting!

🧠 How It Works

Embedding: When you upload a file, the sentence-transformers model converts all your 'Question' text into numbers (vectors).

Vector Search: When you ask a question, the bot converts your question into numbers as well.

Cosine Similarity: It compares your question's numbers against the file's numbers to find the closest match.

Thresholding: If the match score is higher than your Confidence Threshold (e.g., 0.6), it shows the answer. If not, it admits it doesn't know.

⚠️ Troubleshooting

"File must contain..." Error: Ensure your headers are exactly named Question and Answer (case-sensitive).

Slow Initial Load: The first time you run this, it will download the AI model (approx 80MB). Subsequent runs will be instant.