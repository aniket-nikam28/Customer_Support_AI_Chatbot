import sqlite_patch
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI support assistant. Reformulate user queries if needed."),
        ("human", "{input}"),
        ("placeholder", "{chat_history}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=history_prompt
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Store."
            "Keep answers concise (2–4 sentences).",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("system", "Context:\n{context}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)


st.set_page_config(page_title="AI Customer Support Bot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.user-msg {
    background-color: #2b8a3e;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    float: right;
    clear: both;
    text-align: right;
}
.assistant-msg {
    background-color: #1c1c1e;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    float: left;
    clear: both;
    text-align: left;
}
.user-avatar {
    float: right;
    margin-left: 8px;
}
.assistant-avatar {
    float: left;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>UrbanCart Customer Support AI Bot</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Your personal assistant for UrbanCart!</h4>", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### 📖 About this bot")
    st.write("This AI bot answers customer support questions using RAG with Gemini and ChromaDB.")
    st.markdown("**Tech stack:** LangChain, Gemini API, ChromaDB, HuggingFace Embeddings")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("🔗 [View on GitHub](https://github.com/aniket-nikam28/Customer_Support_AI_Chatbot)")

def display_message(role, content):
    if role == "user":
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: flex-end; clear: both;">
                <div class="user-msg">{content}</div>
                <div class="user-avatar">👤</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:  
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: flex-start; clear: both;">
                <div class="assistant-avatar">🤖</div>
                <div class="assistant-msg">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  

for role, content in st.session_state.chat_history:
    display_message(role, content)

st.markdown(
    """
    <script>
    var chatContainer = window.parent.document.querySelector('.main');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """,
    unsafe_allow_html=True
)



user_query = st.chat_input("Ask your query here")



if user_query:
    st.session_state.chat_history.append(("user", user_query))
    display_message("user", user_query)

    with st.spinner("Thinking..."):

        response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
        answer = response.get("answer") or response.get("output_text") or ""

    st.session_state.chat_history.append(("assistant", answer))
    display_message("assistant", answer)

    sources = response.get("context", [])
    if sources:
        with st.expander("🔍 Sources used"):
            for i, doc in enumerate(sources, 1):
                snippet = doc.page_content[:200].replace("\n", " ")
                st.markdown(f"**{i}.** {snippet}  \n📄 *{doc.metadata.get('source', 'unknown')}*")





