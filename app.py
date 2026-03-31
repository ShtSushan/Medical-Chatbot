from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os
import time

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatmodel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    groq_api_key=GROQ_API_KEY
)

question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ── Chat History ──


store = {}
session_timestamps = {}

def get_session_history(session_id: str):
    # Clean sessions older than 1 hour
    current_time = time.time()
    expired = [sid for sid, t in session_timestamps.items()
               if current_time - t > 3600]
    for sid in expired:
        store.pop(sid, None)
        session_timestamps.pop(sid, None)

    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    session_timestamps[session_id] = current_time

    history = store[session_id]
    # Sliding window — keep last 10 messages
    if len(history.messages) > 10:
        history.messages = history.messages[-10:]

    return history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

@app.route("/")
def index():
    return render_template('chatbot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.json.get("msg")
    session_id = request.json.get("session_id", "default")
    print(msg)
    response = conversational_rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}}
    )
    print("Response:", response["answer"])
    return jsonify({"response": response["answer"]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)