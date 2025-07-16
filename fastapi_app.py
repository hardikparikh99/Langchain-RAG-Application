from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
import os
import uuid
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Import our utilities from new file structure
from data_models import DocumentUploadResponse, DocumentDeleteResponse, QueryRequest, QueryResponse
from database_vec import initialize_vector_store, index_document_to_vectorstore, delete_doc_from_vectorstore, InMemoryChatHistory
from doc_processor import ALLOWED_EXTENSIONS, DEFAULT_RETRIEVAL_K, load_and_split_document

load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY", "your_default_api_key_here"),  # Retrieve API key from environment variables
    model="llama-3.3-70b-versatile"
)

# Initialize Vector Store
try:
    vectorstore = initialize_vector_store()
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}")
    raise

# Create FastAPI app
app = FastAPI(
    title="RAG Document Query Application",
    description="API for indexing documents and answering questions using RAG Application",
    version="1.0.0",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "persistAuthorization": True
    }
)

# Create a temporary directory for file uploads
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configuration settings (moved from config.py)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# --- API Endpoints ---

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None)
):
    try:
        # Check if file extension is allowed
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Generate a unique document ID
        document_id = str(uuid.uuid4())
        
        # Use filename as title if not provided
        if title is None:
            title = file.filename
        
        # Create a predictable filename for storage
        stored_filename = f"{document_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, stored_filename)
        
        # Save the file permanently
        with open(file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        logger.info(f"Uploaded file {file.filename} saved to {file_path}")
        
        # Get the approximate number of chunks that will be created
        try:
            chunks = load_and_split_document(file_path)
            chunk_count = len(chunks)
        except Exception as e:
            logger.warning(f"Could not pre-count chunks: {str(e)}")
            chunk_count = 0
        
        # Process the document in the background
        background_tasks.add_task(
            index_document_to_vectorstore,
            vectorstore,
            file_path,
            document_id,
            title
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            title=title,
            filename=file.filename,  # Add original filename to response
            status="processing",
            message=f"Document uploaded and being processed.",
            chunk_count=chunk_count
        )
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Get chat history from in-memory store
        history = InMemoryChatHistory.get_history(request.conversation_id)
        
        logger.info(f"Processing query: '{request.query}' with k={request.k}")
        
        # Check for explicit file type mentions
        file_type_mentions = {
            'excel': ['excel', 'xlsx', 'xls', 'spreadsheet', 'sheet', 'table', 'row', 'column', 'cell'],
            'powerpoint': ['powerpoint', 'ppt', 'pptx', 'slides', 'presentation', 'slide', 'deck'],
            'pdf': ['pdf', 'document', 'page'],
            'word': ['word', 'docx', 'doc', 'document']
        }
        
        query_lower = request.query.lower()
        mentioned_file_types = []
        for file_type, keywords in file_type_mentions.items():
            if any(keyword in query_lower for keyword in keywords):
                mentioned_file_types.append(file_type)
                logger.info(f"Query mentions {file_type}")
                
        # Retrieve with increased k
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": max(request.k * 4, 12),  # Get more docs for filtering
            }
        )
        
        # Get relevant documents
        relevant_docs = retriever.get_relevant_documents(request.query)
        
        logger.info(f"Retrieved {len(relevant_docs)} documents")
        
        # Process documents for diversity
        docs_by_type = {}
        for doc in relevant_docs:
            doc_type = doc.metadata.get('file_type', 'unknown')
            if doc_type not in docs_by_type:
                docs_by_type[doc_type] = []
            docs_by_type[doc_type].append(doc)
        
        logger.info(f"Document types: {[(k, len(v)) for k, v in docs_by_type.items()]}")
        
        # If specific file types are mentioned, prioritize them
        final_docs = []
        
        # First add documents from mentioned file types
        if mentioned_file_types:
            for mentioned_type in mentioned_file_types:
                for doc_type in list(docs_by_type.keys()):
                    if mentioned_type in doc_type and docs_by_type[doc_type]:
                        # Take up to half the requested docs from mentioned types
                        count_to_take = min(len(docs_by_type[doc_type]), max(2, request.k // 2))
                        final_docs.extend(docs_by_type[doc_type][:count_to_take])
                        docs_by_type[doc_type] = docs_by_type[doc_type][count_to_take:]
        
        # Then ensure diversity from all available types
        target_k = request.k
        
        # Calculate how many docs to take from each type to ensure diversity
        remaining_slots = target_k - len(final_docs)
        if remaining_slots > 0 and docs_by_type:
            docs_per_type = max(1, remaining_slots // len(docs_by_type))
            
            # Take docs_per_type from each file type
            for doc_type, docs in list(docs_by_type.items()):
                if docs:
                    count_to_take = min(len(docs), docs_per_type)
                    final_docs.extend(docs[:count_to_take])
                    docs_by_type[doc_type] = docs[count_to_take:]
            
            # If we still have slots, fill them with round-robin
            remaining_slots = target_k - len(final_docs)
            while remaining_slots > 0 and any(len(docs) > 0 for docs in docs_by_type.values()):
                for doc_type in list(docs_by_type.keys()):
                    if docs_by_type[doc_type]:
                        final_docs.append(docs_by_type[doc_type].pop(0))
                        remaining_slots -= 1
                        if remaining_slots == 0:
                            break
        
        # If we somehow ended up with no documents, fall back to the original list
        if not final_docs and relevant_docs:
            final_docs = relevant_docs[:target_k]
            
        relevant_docs = final_docs
        
        # Format documents for context with file type information
        if relevant_docs:
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                file_type = doc.metadata.get('file_type', 'Unknown').upper()
                title = doc.metadata.get('title', 'Unknown')
                page = doc.metadata.get('page') or doc.metadata.get('page_number', 'Unknown')
                
                # Create context with explicit file type markers
                context_part = (
                    f"[DOCUMENT {i+1}]\n"
                    f"Document Type: {file_type}\n"
                    f"Document Title: {title}\n"
                    f"Page/Slide: {page}\n"
                    f"Content: {doc.page_content}"
                )
                context_parts.append(context_part)
            
            context = "\n\n" + "\n\n".join(context_parts)
        else:
            context = "No relevant documents found in the database."
            
        # Create improved system prompt that emphasizes document type awareness
        system_prompt = """You are a document assistant that answers questions based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
1. Only use information that is explicitly present in the context provided below.
2. Pay special attention to the document type (PDF, EXCEL, PPT, etc.) and respond with content from the specific document type the user is asking about.
3. If the user explicitly mentions or asks about PowerPoint, Excel, or PDF, prioritize information from that document type.
4. If information from multiple document types is available and relevant, explicitly mention which information comes from which document type.
5. If the answer isn't in the provided context, say "I don't have information about this in the uploaded documents." Don't make up answers.
6. Always provide document names, types, and page numbers as sources for your information.
7. If the user mentions "slides" or "presentation," look specifically for PPT/PPTX content in the context.
8. If the user mentions "spreadsheet," "cells," or similar, look specifically for Excel/XLS/XLSX content.
9. Maintain accuracy - don't confuse information between different documents.

Context: {context}"""

        # Create the prompt template with improved instructions
        prompt_messages = [("system", system_prompt)]
        
        # Add history messages to the prompt
        for message in history:
            prompt_messages.append((message["type"], message["content"]))
        
        # Add the current question
        prompt_messages.append(("human", "{question}"))
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        
        # Create the chain
        chain = prompt | llm
        
        # Get the response
        response = chain.invoke({
            "context": context,
            "question": request.query
        })
        
        # Update in-memory chat history
        InMemoryChatHistory.add_message(request.conversation_id, "human", request.query)
        InMemoryChatHistory.add_message(request.conversation_id, "ai", response.content)
        
        # Prepare source information
        sources = [
            {
                "title": doc.metadata.get("title", "Unknown"),
                "page": doc.metadata.get("page") or doc.metadata.get("page_number", "Unknown"),
                "document_id": doc.metadata.get("document_id", "Unknown"),
                "file_type": doc.metadata.get("file_type", "Unknown").upper(),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for doc in relevant_docs
        ]
        
        return QueryResponse(
            answer=response.content,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Fix: Add a separate endpoint for GET /documents that doesn't require a document_id
@app.get("/documents")
async def get_all_documents():
    """
    Endpoint to retrieve all documents from the vector store.
    
    Returns:
        A list of all documents with their metadata.
    """
    try:
        # We need to query the vector store to get all documents
        # This is a simplified version - in a real app you'd want pagination
        # and filtering capabilities
        
        # Dummy query to get all documents - not ideal but works for demonstration
        dummy_query = "document"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 100})  # Get up to 100 docs
        docs = retriever.get_relevant_documents(dummy_query)
        
        # Extract unique document IDs and metadata
        unique_docs = {}
        for doc in docs:
            doc_id = doc.metadata.get("document_id")
            if doc_id and doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    "document_id": doc_id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "filename": doc.metadata.get("filename", doc.metadata.get("title", "Unknown")),  # Add filename to response
                    "created_at": doc.metadata.get("created_at", "Unknown"),
                    "status": "indexed",  # All retrieved docs are indexed
                    "chunk_count": 1  # Initialize chunk count
                }
            elif doc_id:
                # Increment chunk count for existing document
                unique_docs[doc_id]["chunk_count"] += 1
        
        # Convert to list
        document_list = list(unique_docs.values())
        
        return {
            "documents": document_list,
            "count": len(document_list)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        # Even if there's an error, return an empty list instead of raising an exception
        # This ensures the frontend doesn't break
        return {
            "documents": [],
            "count": 0
        }

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Endpoint to get specific document details.
    """
    try:
        # Implement specific document retrieval logic here
        # For now, we'll use the same vectorstore query but filter by document_id
        dummy_query = "document"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
        docs = retriever.get_relevant_documents(dummy_query)
        
        # Filter docs by document_id
        filtered_docs = [doc for doc in docs if doc.metadata.get("document_id") == document_id]
        
        if not filtered_docs:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {document_id} not found"
            )
                
        # Extract metadata from the first chunk
        doc_metadata = filtered_docs[0].metadata
        
        return {
            "document_id": document_id,
            "title": doc_metadata.get("title", "Unknown"),
            "filename": doc_metadata.get("filename", doc_metadata.get("title", "Unknown")),
            "created_at": doc_metadata.get("created_at", "Unknown"),
            "status": "indexed",
            "chunk_count": len(filtered_docs)
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document details: {str(e)}"
        )

@app.get("/chat-history")
async def get_chat_history():
    """
    Endpoint to retrieve a list of all conversation histories with metadata.
    
    Returns:
        A list of all conversation IDs with timestamps and titles
    """
    try:
        # Get all chat histories
        histories = InMemoryChatHistory.get_all_histories()
        
        return {
            "conversations": histories,
            "count": len(histories)
        }
    except Exception as e:
        logger.error(f"Error retrieving chat histories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat histories: {str(e)}"
        )

@app.get("/chat-history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """
    Endpoint to retrieve the chat history for a specific conversation.
    
    Args:
        conversation_id: The unique identifier for the conversation
        
    Returns:
        The chat history as a list of messages with type and content,
        or an empty list if no history exists
    """
    try:
        # Get chat history from in-memory store
        history = InMemoryChatHistory.get_history(conversation_id)
        
        # Return formatted history
        return {
            "conversation_id": conversation_id,
            "message_count": len(history),
            "messages": history
        }
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )

@app.delete("/conversations/{conversation_id}")
async def delete_conversation_history(conversation_id: str):
    try:
        # Validate conversation ID exists before clearing
        history = InMemoryChatHistory.get_history(conversation_id)
        
        if not history:
            # If there's no history, return a different status but don't raise an exception
            return {"status": "not_found", "message": f"No conversation history found for ID: {conversation_id}"}
        
        # Get message count before clearing for informative response
        message_count = len(history)
        
        # Clear conversation history using the InMemoryChatHistory class method
        InMemoryChatHistory.clear_history(conversation_id)
        
        logger.info(f"Cleared conversation history for ID: {conversation_id} ({message_count} messages)")
        
        return {
            "status": "success", 
            "message": f"Conversation {conversation_id} history cleared",
            "messages_removed": message_count
        }
    except Exception as e:
        logger.error(f"Error deleting conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting conversation history: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)