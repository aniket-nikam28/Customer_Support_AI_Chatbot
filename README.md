#  UrbanCart Customer Support AI Bot

An AI-powered **customer support  AI chatbot** built with **LangChain, Google Gemini, ChromaDB, HuggingFace Embeddings, and Streamlit**.  

This bot uses **Retrieval-Augmented Generation (RAG)** to answer customer queries based on UrbanCart’s knowledge base.  
It supports **multi-turn conversations with memory** and features a **modern chat-style UI**.

---

## Features

- **AI-Powered Answers** – Uses Google Gemini API with LangChain.  
- **RAG (Retrieval-Augmented Generation)** – Fetches answers from stored documents (PDF, TXT, CSV).  
- **Conversation Memory** – Maintains context across multiple queries.  
- **Chat-Style UI** – WhatsApp-like interface with left/right aligned messages and avatars.  
- **Fast & Lightweight** – Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.  

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/) – Orchestration framework  
- [Google Gemini API](https://ai.google.dev/) – LLM backend  
- [ChromaDB](https://www.trychroma.com/) – Vector database  
- [HuggingFace Sentence Transformers](https://www.sbert.net/) – Embeddings  
- [Streamlit](https://streamlit.io/) – Web UI  

---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/aniket-nikam28/Customer_Support_AI_Chatbot.git
cd Customer_Support_AI_Chatbot

### 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add environment variables

Create a .env file in the root directory:

GOOGLE_API_KEY=your_google_gemini_api_key
CHROMA_PERSIST_DIR=./chroma_db

### 5. Prepare the knowledge base

Put your documents in the data/ folder (PDF, TXT, CSV supported).
Then run:

python ingestion.py


This will process the documents, split them into chunks, and store embeddings in ChromaDB.

### 6. Run the chatbot
streamlit run app.py


Your bot will be live at: http://localhost:8501

## Deployment
Deploy on Streamlit Cloud

Push your repo to GitHub.

Go to Streamlit Cloud
.

Connect your GitHub repo.

Set up environment variables (GOOGLE_API_KEY, CHROMA_PERSIST_DIR).

Deploy 

## Screenshots
Chat UI

User messages (green, right) and AI responses (dark, left) with avatars.


## License

This project is licensed under the MIT License.

## Author

Aniket Nikam
🔗 LinkedIn

