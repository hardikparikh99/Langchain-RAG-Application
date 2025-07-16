import os
import time
from datetime import datetime
import json
import streamlit as st
import requests
import uuid
import pandas as pd

# Constants
API_URL = "http://127.0.0.1:8000"  # FastAPI server URL
ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.ppt', '.xlsx', '.xls', '.html', '.txt']

# Page Configuration
st.set_page_config(
    page_title="RAG Document Query Application",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-container {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .doc-upload {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .doc-table {
        margin-top: 20px;
    }
    .sources-expander {
        background-color: #f9f9f9;
        border-radius: 5px;
    }
    .main-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-header {
        margin-top: 15px;
        margin-bottom: 5px;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .user-message {
        background-color: #e0f7fa;
        text-align: right;
    }
    .assistant-message {
        background-color: #f1f3f4;
    }
    .history-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f9f9f9;
        cursor: pointer;
    }
    .history-item:hover {
        background-color: #e0f7fa;
    }
    .document-item {
        padding: 5px 0;
        border-bottom: 1px solid #eee;
    }
    .document-title {
        font-weight: bold;
    }
    .document-filename {
        color: #666;
        font-size: 0.9em;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history_list" not in st.session_state:
    st.session_state.chat_history_list = []

# Helper Functions
def query_documents(query, conversation_id, k=5):  # Increased default k
    """Send a query to the FastAPI endpoint with improved parameters"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query, 
                "conversation_id": conversation_id, 
                "k": k,
                "use_only_provided_context": True,
                "filter_threshold": 0.5,  # Lower threshold for better retrieval across doc types
                "ensure_document_diversity": True  # New parameter to signal we want diverse docs
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # Error handling code stays the same
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', '')
                error_msg = f"{error_msg}: {error_detail}"
            except:
                pass
        st.error(f"Error querying documents: {error_msg}")
        return None

def upload_document(file):
    """Upload a document to the FastAPI endpoint"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        
        response = requests.post(
            f"{API_URL}/documents/upload",
            files=files
        )
        response.raise_for_status()
        result = response.json()
        
        # Return the full result including chunk_count if available
        return result
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', '')
                error_msg = f"{error_msg}: {error_detail}"
            except:
                pass
        st.error(f"Error uploading document: {error_msg}")
        return None

def get_chat_history(conversation_id):
    """Retrieve chat history for a conversation"""
    try:
        response = requests.get(f"{API_URL}/chat-history/{conversation_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', '')
                error_msg = f"{error_msg}: {error_detail}"
            except:
                pass
        st.error(f"Error retrieving chat history: {error_msg}")
        return {"messages": []}

def clear_conversation(conversation_id):
    """Clear the conversation history"""
    try:
        response = requests.delete(f"{API_URL}/conversations/{conversation_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', '')
                error_msg = f"{error_msg}: {error_detail}"
            except:
                pass
        st.error(f"Error clearing conversation: {error_msg}")
        return None

def get_file_extension(filename):
    """Get file extension from filename in lowercase"""
    return os.path.splitext(filename)[1].lower()

def auto_save_chat_history():
    """Automatically save the chat history if there are messages"""
    if st.session_state.messages:
        filename = save_chat_history_to_file(
            st.session_state.conversation_id, 
            st.session_state.messages
        )
        return filename
    return None

def save_chat_history_to_file(conversation_id, messages):
    """Save chat history to a JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs("chat_history", exist_ok=True)
        
        # Create a filename with conversation_id
        # Using just the conversation_id ensures we overwrite the same file
        # instead of creating multiple files per conversation
        filename = f"chat_history/conversation_{conversation_id}.json"
        
        # Prepare the data to save
        history_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": messages
        }
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
            
        return filename
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")
        return None

def load_chat_histories():
    """Load all available chat histories from the chat_history directory"""
    histories = []
    try:
        if not os.path.exists("chat_history"):
            os.makedirs("chat_history", exist_ok=True)
            return []
            
        for filename in os.listdir("chat_history"):
            if filename.endswith(".json") and filename.startswith("conversation_"):
                filepath = os.path.join("chat_history", filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    timestamp = datetime.fromisoformat(data["timestamp"])
                    # Get first few messages for preview
                    message_preview = ""
                    if data["messages"]:
                        message_preview = data["messages"][0]["content"]
                        if len(message_preview) > 50:
                            message_preview = message_preview[:50] + "..."
                    
                    histories.append({
                        "conversation_id": data["conversation_id"],
                        "timestamp": timestamp,
                        "timestamp_str": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "message_count": len(data["messages"]),
                        "preview": message_preview,
                        "filename": filepath
                    })
                    
        # Sort by timestamp, newest first
        histories.sort(key=lambda x: x["timestamp"], reverse=True)
        return histories
    except Exception as e:
        st.error(f"Error loading chat histories: {str(e)}")
        return []

def load_selected_chat_history(conversation_id):
    """Load a specific chat history by conversation ID"""
    try:
        filepath = f"chat_history/conversation_{conversation_id}.json"
        if not os.path.exists(filepath):
            st.error(f"Chat history file not found: {filepath}")
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data["messages"]
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return None

def delete_chat_history(conversation_id):
    """Delete a specific chat history file"""
    try:
        filepath = f"chat_history/conversation_{conversation_id}.json"
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        else:
            st.error(f"Chat history file not found: {filepath}")
            return False
    except Exception as e:
        st.error(f"Error deleting chat history: {str(e)}")
        return False

def get_all_documents():
    """Retrieve all documents from the backend database"""
    try:
        response = requests.get(f"{API_URL}/documents")
        response.raise_for_status()
        response_data = response.json()
        
        # Check if the response has the expected structure
        if isinstance(response_data, dict) and "documents" in response_data:
            documents = response_data["documents"]
            
            # Format the documents for the UI
            formatted_docs = []
            for doc in documents:
                formatted_docs.append({
                    "document_id": doc.get("document_id", "unknown"),
                    "title": doc.get("title", "No title"),
                    "filename": doc.get("filename", "Unknown filename"),
                    "status": doc.get("status", "processing"),
                    "upload_time": doc.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "chunks": doc.get("chunk_count", 0),
                    "download_url": f"{API_URL}/documents/{doc.get('document_id', 'unknown')}/file"
                })
            
            return formatted_docs
        else:
            # If response doesn't have expected structure
            st.warning(f"Unexpected API response format: {response_data}")
            return []
            
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', '')
                error_msg = f"{error_msg}: {error_detail}"
            except:
                pass
        st.error(f"Error retrieving documents: {error_msg}")
        return []

def check_if_documents_exist():
    """Check if there are any documents in the database"""
    try:
        docs = get_all_documents()
        return len(docs) > 0
    except:
        # If there's an error, assume documents might exist
        return True

def restore_chat_history(history):
    """Restore a selected chat history"""
    # Auto-save current conversation before switching
    auto_save_chat_history()
    
    # Load the selected history
    messages = load_selected_chat_history(history['conversation_id'])
    if messages:
        # Update session state
        st.session_state.conversation_id = history['conversation_id']
        st.session_state.messages = messages
        
        # Clear backend conversation (just to be safe)
        clear_conversation(history['conversation_id'])
        
        # Show success message
        return True
    return False

# UI Components
def sidebar():
    with st.sidebar:
        st.title("📚 RAG Document Query Application")
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Document Upload Section
        st.markdown("### 📤 Upload Documents")
        
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Choose a document", 
                type=["pdf", "docx", "pptx", "ppt", "xlsx", "xls", "html", "txt"]
            )
            
            supported_formats = ", ".join([ext[1:] for ext in ALLOWED_EXTENSIONS])
            st.caption(f"Supported formats: {supported_formats}")
            
            if uploaded_file is not None:
                if st.button("📤 Upload Document", type="primary"):
                    # Check file extension
                    extension = get_file_extension(uploaded_file.name)
                    if extension not in ALLOWED_EXTENSIONS:
                        st.error(f"Unsupported file type. Allowed types: {supported_formats}")
                    else:
                        with st.spinner("📄 Processing document..."):
                            result = upload_document(uploaded_file)
                            if result:
                                chunk_count = result.get("chunk_count", 0)
                                
                                # Show popup with chunk count information
                                if chunk_count > 0:
                                    st.success(f"✅ Document uploaded successfully!")
                                    # Use a popup to display chunk information
                                    st.info(f"📊 {chunk_count} chunks created.")
                                else:
                                    st.success(f"✅ Document uploaded successfully!")
        
        # Document Management Section - Shows only original filenames
        st.markdown("### 📁 Document Management")

        with st.container(border=True):
            docs = get_all_documents()
            if docs:
                st.info(f"🗃️ {len(docs)} documents in database")
                    
                # Display only original filenames
                for doc in docs:
                    filename = doc['filename']  # Directly use the filename field
                    st.markdown(f"📄 {filename}")
            else:
                st.info("🚫 No documents uploaded yet.")
        
        # Conversation Management
        st.markdown("### 🗨️ Conversation")
        
        with st.container(border=True):
            # Only keep the Start New Conversation button
            if st.button("🆕 Start New Conversation", type="primary"):
                # Auto-save current chat before clearing
                auto_save_chat_history()
                
                # Create new conversation
                st.session_state.conversation_id = str(uuid.uuid4())
                st.session_state.messages = []
                clear_conversation(st.session_state.conversation_id)
                st.success("✅ Started new conversation!")
                
            st.caption(f"Current Conversation ID: `{st.session_state.conversation_id}`")

        # Chat History Section - Now directly in sidebar as a stack
        st.markdown("### 🕘 Chat History")

        with st.container(border=True):
            # Refresh chat history list on sidebar render
            if "chat_history_list" not in st.session_state or not st.session_state.chat_history_list:
                st.session_state.chat_history_list = load_chat_histories()
                
            if not st.session_state.chat_history_list:
                st.info("No saved chat histories found.")
            else:
                # Display chat histories as a stack in sidebar
                for i, history in enumerate(st.session_state.chat_history_list):
                    # Create a clickable container for each history item
                    history_container = st.container(border=True)
                    with history_container:
                        # Make the entire container clickable for restoring the history
                        col1, col2 = st.columns([9, 1])
                        
                        with col1:
                            # Create a button that looks like text for the history preview
                            if st.button(f"**{history['preview']}**\n{history['timestamp_str']}", 
                                         key=f"restore_{i}",
                                         help="Click to restore this conversation"):
                                if restore_chat_history(history):
                                    st.success("✅ Chat history restored!")
                                    st.rerun()
                        
                        with col2:
                            if st.button("❌", key=f"delete_{i}", help="Delete this conversation"):
                                if delete_chat_history(history['conversation_id']):
                                    st.session_state.chat_history_list.pop(i)
                                    st.success(f"✅ Deleted")
                                    st.rerun()

def display_chat():
    st.markdown("<h1 class='main-header'>🤖 Document Q&A Assistant</h1>", unsafe_allow_html=True)
    
    # Create chat container with border
    chat_container = st.container(border=True)
    
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Welcome! Upload your documents and ask questions about them. The AI will find relevant information from your documents.")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(message["content"])
            else:  # assistant
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Source References"):
                            for i, source in enumerate(message["sources"]):
                                file_type = source.get('file_type', 'Unknown')
                                st.markdown(f"**Source {i+1}:** {source['title']} ({file_type}, Page {source['page']})")
                                st.markdown(f"*Snippet:* {source['snippet']}")
    
        # Chat input with placeholder
        query = st.chat_input("Ask a question about your documents...")
         
        if query:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.write(query)
            
            # Get AI response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Searching through documents..."):
                    # Create query with default parameters
                    enhanced_query = {
                        "query": query,
                        "conversation_id": st.session_state.conversation_id,
                        "k": 5,  # Using default value of 5
                        "ensure_document_diversity": True
                    }
                    
                    # Send the query to the backend
                    response = requests.post(f"{API_URL}/query", json=enhanced_query)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        st.write(response_data["answer"])
                        
                        # Display sources
                        if response_data["sources"]:
                            with st.expander("📚 Source References"):
                                for i, source in enumerate(response_data["sources"]):
                                    file_type = source.get('file_type', 'Unknown')
                                    st.markdown(f"**Source {i+1}:** {source['title']} ({file_type}, Page {source['page']})")
                                    st.markdown(f"*Snippet:* {source['snippet']}")
                        
                        # Save assistant message with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_data["answer"],
                            "sources": response_data["sources"]
                        })
                        
                        # Auto-save chat history
                        auto_save_chat_history()
                        st.session_state.chat_history_list = load_chat_histories()
                    else:
                        # Define error_message first before using it
                        error_message = f"❌ Failed to get a response. Status code: {response.status_code}"
                        try:
                            error_detail = response.json().get('detail', '')
                            error_message += f" - {error_detail}"
                        except:
                            pass
                        
                        # Now use error_message after it's defined
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_message,
                            "sources": []
                        })

def main():
    sidebar()
    display_chat()

if __name__ == "__main__":
    main()