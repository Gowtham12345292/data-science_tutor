import streamlit as st
import sqlite3
import uuid
import time
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import io

# Load API key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Set up the Gemini 1.5 Pro model
llm = GoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-pro")

# Initialize SQLite database
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Function to save messages
def save_message(session_id, role, content):
    cursor.execute("INSERT INTO chat (session_id, role, content) VALUES (?, ?, ?)", 
                  (session_id, role, content))
    conn.commit()

# Function to load chat history
def load_chat_history(session_id):
    cursor.execute("SELECT role, content, timestamp FROM chat WHERE session_id = ? ORDER BY timestamp", 
                  (session_id,))
    return cursor.fetchall()

# Chat history instance
def chat_history(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"
    )

# Generate unique session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Custom CSS
st.markdown("""
    <style>
        .title-text {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .stTextInput {
            position: fixed;
            bottom: 60px;
            width: 70%;
            left: 15%;
            z-index: 999;
        }
        .code-block {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
        }
        .timestamp {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for additional controls
with st.sidebar:
    st.title("Chat Controls")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
            <style>
                .stApp {
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                }
                .code-block {
                    background-color: #2D2D2D;
                    color: #FFFFFF;
                }
            </style>
        """, unsafe_allow_html=True)
    
    if st.button("🆕 New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # Export chat feature
    if st.button("📥 Export Chat"):
        chat_data = load_chat_history(st.session_state.session_id)
        export_text = "\n\n".join([f"{row[0].upper()}[{row[2]}]: {row[1]}" 
                                 for row in chat_data])
        st.download_button(
            label="Download Chat",
            data=export_text,
            file_name=f"chat_{st.session_state.session_id}.txt",
            mime="text/plain"
        )

# Animated Title Function
def animated_text(text, speed=0.03):
    placeholder = st.empty()
    displayed_text = ""
    for letter in text:
        displayed_text += letter
        placeholder.markdown(f"""
            <h1 class="title-text" style="color: {'#00D1FF' if theme == 'Light' else '#FFD700'}">
                {displayed_text} 🚀
            </h1>
        """, unsafe_allow_html=True)
        time.sleep(speed)

# Display Animated Welcome Message
animated_text('Data Science Tutor AI')

# Get session ID and chat history
session_id = st.session_state.session_id
chat_history_instance = chat_history(session_id)

# Define Chat Prompt Template
chat_prompt = ChatPromptTemplate(
    messages=[
        ('system', """You are an AI assistant specialized in Data Science tutoring. 
                      You will only answer questions related to Data Science. 
                      Provide code examples with proper syntax highlighting when relevant.
                      If asked anything outside this topic, politely decline and request a Data Science-related question.
                   """),
        MessagesPlaceholder(variable_name="history", optional=True),
        ('human', '{prompt}')
    ]
)

# Create chain
out_parser = StrOutputParser()
chain = chat_prompt | llm | out_parser
chat = RunnableWithMessageHistory(
    chain,
    lambda session: SQLChatMessageHistory(session, "sqlite:///chat_history.db"),
    input_messages_key="prompt",
    history_messages_key="history"
)

# Chat History Container
chat_container = st.container()

# Load and display chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history(session_id)

with chat_container:
    for role, content, timestamp in st.session_state.messages:
        with st.chat_message(role):
            # Handle code blocks in content
            if "```" in content:
                parts = content.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        st.markdown(part)
                    else:
                        st.markdown(f'<div class="code-block">{part}</div>', 
                                  unsafe_allow_html=True)
            else:
                st.markdown(content)
            st.markdown(f'<div class="timestamp">{timestamp}</div>', 
                       unsafe_allow_html=True)

# User input with additional features
with st.form(key='chat_form', clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Ask a Data Science question:", key="user_message")
    with col2:
        submit_button = st.form_submit_button("Send")

# Process user input
if submit_button and user_input:
    save_message(session_id, "user", user_input)
    st.session_state.messages.append(("user", user_input, time.strftime('%Y-%m-%d %H:%M:%S')))

    config = {'configurable': {'session_id': session_id}}
    response = chat.invoke({'prompt': user_input}, config)

    save_message(session_id, "assistant", response)
    st.session_state.messages.append(("assistant", response, time.strftime('%Y-%m-%d %H:%M:%S')))
    st.rerun()

# Help button in bottom right
st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px;">
        <button onclick="alert('Ask me anything about Data Science! Examples: Python, SQL, Machine Learning, Statistics')">ℹ️ Help</button>
    </div>
""", unsafe_allow_html=True)
