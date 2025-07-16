from datetime import time
import os
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import logging
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration settings (moved from config.py)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "langchain-rag")

def initialize_vector_store():
    """Initialize and return a vector store instance with improved retrieval options"""
    try:
        # Initialize Pinecone
        pc = PineconeClient(
            api_key=PINECONE_API_KEY
        )
        
        # Initialize embedding function with a more robust model if possible
        embedding_function = OllamaEmbeddings(model="llama3.2:1b")
        
        # Check if index exists, if not create it
        existing_indexes = pc.list_indexes().names()
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=2048,  # Adjust dimension based on your embedding model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Initialize vector store with hybrid search if possible
        index = pc.Index(INDEX_NAME)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_function,
            text_key="text",
            namespace="documents",
        )
        logger.info("Successfully initialized Pinecone vector store")
        return vectorstore
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        raise

def index_document_to_vectorstore(vectorstore, file_path: str, document_id: str, title: str) -> bool:
    """
    Index a document to the vector store with explicit file type metadata
    """
    from doc_processor import load_and_split_document
    
    try:
        # Get file extension for metadata
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension.startswith('.'):
            file_extension = file_extension[1:]  # Remove the dot
        
        # Load and chunk the document
        splits = load_and_split_document(file_path)

        # Add document_id, title and file_type to each split's metadata AND content
        for split in splits:
            split.metadata['document_id'] = document_id
            split.metadata['title'] = title
            split.metadata['file_type'] = file_extension  # Add explicit file type
            split.metadata['file_path'] = file_path  # Store the file path for later retrieval
            
            # Add file type to content for better retrieval
            # This makes the document content itself contain the file type
            split.page_content = f"[Document Type: {file_extension.upper()}]\n{split.page_content}"

        # Add to vectorstore
        vectorstore.add_documents(splits)
        logger.info(f"Successfully indexed {len(splits)} chunks from {file_path} with file_type {file_extension}")
        
        # DO NOT delete the file - keep it for persistent storage
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        #     logger.info(f"Cleaned up temporary file: {file_path}")
            
        return True
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        # DO NOT delete the file even if processing failed
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        #     logger.info(f"Cleaned up temporary file after error: {file_path}")
        return False

async def fetch_all_documents_from_vectorstore(vectorstore):
    """
    Retrieves all unique documents from the Pinecone vector store.
    
    Args:
        vectorstore: The initialized vector store instance
        
    Returns:
        A list of document metadata objects
    """
    try:
        logger.info("Fetching all documents from vector store")
        
        # For Pinecone, we need to query the index to get metadata
        # Using a very generic query to retrieve documents
        results = vectorstore.similarity_search(
            query="",
            k=1000  # Large k to get as many docs as possible
        )
        
        # Extract unique documents based on document_id
        unique_docs = {}
        for doc in results:
            doc_id = doc.metadata.get("document_id")
            if doc_id and doc_id not in unique_docs:
                # Create a clean document object with relevant metadata
                unique_docs[doc_id] = {
                    "document_id": doc_id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "created_at": doc.metadata.get("created_at", doc.metadata.get("timestamp", "Unknown")),
                    "chunk_count": 1,  # Will increment for each chunk
                    "file_type": doc.metadata.get("file_type", doc.metadata.get("source", "Unknown").split(".")[-1] if "." in doc.metadata.get("source", "") else "Unknown"),
                }
            elif doc_id:
                # If we've seen this document before, increment chunk count
                unique_docs[doc_id]["chunk_count"] += 1
        
        # Convert to list for response
        documents_list = list(unique_docs.values())
        
        # Sort by title for consistency
        documents_list.sort(key=lambda x: x["title"])
        
        logger.info(f"Retrieved {len(documents_list)} unique documents from vector store")
        return documents_list
    
    except Exception as e:
        # Log the error and re-raise
        logger.error(f"Error fetching documents from vector store: {str(e)}")
        raise

def delete_doc_from_vectorstore(vectorstore, document_id: str):
    """
    Delete all document chunks with specified document_id from the vector store
    """
    try:
        if isinstance(vectorstore, PineconeVectorStore):
            # Delete from Pinecone
            vectorstore.delete(filter={"document_id": document_id})
        else:
            # Delete from Chroma
            vectorstore._collection.delete(where={"document_id": document_id})
        
        logger.info(f"Deleted all documents with document_id {document_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting document with document_id {document_id}: {str(e)}")
        return False

class InMemoryChatHistory:
    _history_store = {}  # Class variable to store all conversations

    @classmethod
    def get_history(cls, conversation_id):
        """Get or create a chat history for a conversation ID"""
        if conversation_id not in cls._history_store:
            cls._history_store[conversation_id] = []
        return cls._history_store[conversation_id]

    @classmethod
    def add_message(cls, conversation_id, role, content):
        """Add a message to the conversation history"""
        history = cls.get_history(conversation_id)
        history.append({"type": role, "content": content})
        return history

    @classmethod
    def clear_history(cls, conversation_id):
        """Clear the history for a conversation ID"""
        if conversation_id in cls._history_store:
            cls._history_store[conversation_id] = []
        return []
    
    @classmethod
    def get_all_histories(cls):
        """Get all conversation histories with metadata.
        
        Returns:
            list: A list of dictionaries with conversation metadata
        """
        all_histories = []
        for conv_id, messages in cls._history_store.items():
            if messages:
                # Get first message timestamp as conversation start time
                first_message_time = messages[0].get("timestamp", "Unknown")
                
                # Get conversation title - use first query as title or default if empty
                # Find first human message to use as title
                title = "New Conversation"
                for msg in messages:
                    if msg["type"] == "human":
                        # Use first 30 chars of first human message as title
                        title = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
                        break
                
                # Add conversation metadata to list
                all_histories.append({
                    "conversation_id": conv_id,
                    "title": title,
                    "message_count": len(messages),
                    "created_at": first_message_time,
                    "last_updated": messages[-1].get("timestamp", "Unknown")
                })
        
        # Sort by last_updated (most recent first)
        all_histories.sort(key=lambda x: x["last_updated"], reverse=True)
        return all_histories

def get_chat_history(conversation_id: str) -> BaseChatMessageHistory:
    """Get a chat history object for the given conversation ID"""
    redis_url = os.environ.get("REDIS_URL")
    try:
        # If Redis URL is available, use Redis for chat history
        if redis_url:
            return RedisChatMessageHistory(
                conversation_id=conversation_id,
                url=redis_url,
                ttl=60*60*24*7  # 7 days time-to-live
            )
        else:
            logger.warning("Redis URL not found. Falling back to in-memory chat history.")
            return ChatMessageHistory()
    except Exception as e:
        logger.warning(f"Could not initialize Redis chat history: {str(e)}. Falling back to in-memory.")
        # Fall back to in-memory chat history
        return ChatMessageHistory()