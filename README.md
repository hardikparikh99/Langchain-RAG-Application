# Langchain RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain, featuring a FastAPI backend and Streamlit frontend for document processing and question-answering.

## Overview

This project demonstrates how to implement a RAG system using LangChain, which combines retrieval-based and generative approaches to provide context-aware responses. It includes both a FastAPI backend and a Streamlit frontend for flexible deployment options.

## Features

- Document ingestion and chunking
- Vector store integration
- Context-aware response generation
- Semantic search capabilities
- FastAPI REST API
- Streamlit interactive frontend
- Chat history management
- Document upload support

## Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your environment variables

## Running the Application

### FastAPI Backend Server

To run the FastAPI backend server using uvicorn:

```bash
# Development mode with auto-reload
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

# Production mode (no auto-reload)
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

# With multiple workers (production)
# uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

The API server will be available at `http://localhost:8000`

### Streamlit Frontend

To run the Streamlit frontend application:

```bash
# Basic run
streamlit run frontend_app.py

# Or with specific port
# streamlit run frontend_app.py --server.port=8501
```

The frontend will be available at `http://localhost:8501` by default

### API Documentation

Once the FastAPI server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage

### Document Ingestion

1. **Using Streamlit Interface**:
   - Launch the Streamlit frontend
   - Use the document uploader to upload files
   - The system will automatically process and index the documents

2. **Using FastAPI Endpoint**:
   ```python
   import requests
   
   # Upload and process document
   with open("path/to/your/document.pdf", "rb") as f:
       files = {"file": ("document.pdf", f, "application/pdf")}
       response = requests.post("http://localhost:8000/upload/", files=files)
   print(response.json())
   ```

### Querying the System

1. **Using Streamlit Interface**:
   - Enter your question in the chat interface
   - View the response along with source documents

2. **Using FastAPI Endpoint**:
   ```python
   import requests
   
   # Query the system
   response = requests.post(
       "http://localhost:8000/query/",
       json={
           "query": "What is the main topic discussed in the documents?",
           "chat_history": []  # Optional: include previous messages for context
       }
   )
   print(response.json())
   ```

### Environment Variables

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=text-embedding-3-small  # or your preferred embedding model
MODEL_NAME=gpt-3.5-turbo  # or your preferred LLM model
```

## Project Structure

```
Langchain-RAG-Application/
├── .env                    # Environment variables
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
├── __init__.py             # Package initialization
├── __pycache__/            # Python bytecode cache
├── chat_history/           # Directory for storing chat history
├── data_models.py          # Data models and schemas
├── database_vec.py         # Vector database operations and management
├── doc_processor.py        # Document processing and chunking logic
├── fastapi_app.py          # FastAPI application and endpoints
├── frontend_app.py         # Streamlit frontend application
├── requirements.txt        # Python dependencies
└── uploads/                # Directory for uploaded documents
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain documentation
- OpenAI API documentation
- FastAPI and Streamlit documentation
- Various RAG implementation examples