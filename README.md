# RAG (Retrieval-Augmented Generation) Task

A project implementing Retrieval-Augmented Generation using LangChain, with both FastAPI and Streamlit interfaces.

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

### FastAPI Server

To run the FastAPI server using uvicorn:
```bash
# Development mode (auto-reload)
uvicorn fastapi_app:app --reload

# Production mode
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```
The server will be available at `http://localhost:8000`

### Streamlit Frontend

To run the Streamlit frontend:
```bash
streamlit run frontend_app.py
```
The frontend will be available at `http://localhost:8501`

### API Documentation

Once the FastAPI server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage

### Document Ingestion

Upload documents through the Streamlit interface or use the FastAPI endpoint:
```python
from doc_processor import DocumentProcessor

doc_processor = DocumentProcessor()
doc_processor.process_directory("path/to/documents")
```

### Querying

Use the Streamlit interface or make requests to the FastAPI endpoint:
```python
# FastAPI example
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is the main topic discussed in the documents?"}
)
print(response.json())
```

## Project Structure

```
rag_task-1/
├── .env                     # Environment variables
├── __init__.py              # Package initialization
├── data_models.py           # Data models and schemas
├── database_vec.py          # Vector database operations
├── doc_processor.py         # Document processing logic
├── fastapi_app.py           # FastAPI backend server
├── frontend_app.py          # Streamlit frontend application
├── __pycache__/             # Python bytecode cache
├── chat_history/            # Chat history storage
├── uploads/                 # Document upload directory
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
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