import os
import logging
import copy
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    TextLoader
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration settings (moved from config.py)
DEFAULT_CHUNK_SIZE = int(os.environ.get("DEFAULT_CHUNK_SIZE", 2000))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("DEFAULT_CHUNK_OVERLAP", 200))
DEFAULT_RETRIEVAL_K = int(os.environ.get("DEFAULT_RETRIEVAL_K", 3))
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.ppt', '.xlsx', '.xls', '.html', '.htm', '.txt'}

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE, 
    chunk_overlap=DEFAULT_CHUNK_OVERLAP, 
    length_function=len
)

def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metadata keys to ensure consistency across different document loaders
    """
    normalized = {}
    
    # Normalize page/page_number
    if 'page' in metadata:
        normalized['page'] = metadata['page']
    if 'page_number' in metadata:
        normalized['page'] = metadata['page_number']
    
    # Preserve other keys
    for key, value in metadata.items():
        if key not in ['page', 'page_number'] or key not in normalized:
            normalized[key] = value
    
    return normalized

def preserve_minimal_metadata(document):
    """
    Preserve only relevant metadata, remove other metadata
    Add file_type based on file extension if missing
    """
    minimal_doc = copy.deepcopy(document)
    
    # Keep only allowed keys
    allowed_keys = ['page', 'page_number', 'source', 'file_id', 'title', 'document_id']
    
    # Create a new metadata dictionary with only allowed keys
    minimal_metadata = {
        key: minimal_doc.metadata.get(key) 
        for key in allowed_keys 
        if key in minimal_doc.metadata and minimal_doc.metadata[key] is not None
    }
    
    # Add file_type based on source file extension if available
    if 'source' in minimal_metadata and isinstance(minimal_metadata['source'], str):
        file_extension = os.path.splitext(minimal_metadata['source'])[1].lower()
        if file_extension:
            minimal_metadata['file_type'] = file_extension[1:]  # Remove the dot
    
    # Normalize the metadata
    minimal_metadata = normalize_metadata(minimal_metadata)
    
    minimal_doc.metadata = minimal_metadata
    return minimal_doc

def chunk_page(page, chunk_size=2000, chunk_overlap=200):
    """
    Chunk a document page with specified size and overlap
    """
    local_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunked_docs = local_text_splitter.split_documents([page])
    return [preserve_minimal_metadata(chunk) for chunk in chunked_docs]

def pdf_chunking(file_path):
    """Load and chunk a PDF file to create exactly one chunk per page"""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_chunks = []
        page_count = len(documents)
        logger.info(f"PDF loaded: {file_path} with {page_count} pages")
       
        for page in documents:
            # Ensure page number is set consistently
            if 'page' not in page.metadata and 'page_number' in page.metadata:
                page.metadata['page'] = page.metadata['page_number']
            elif 'page' not in page.metadata and 'page_number' not in page.metadata:
                # If neither page key exists, use an index (not ideal but prevents errors)
                try:
                    page.metadata['page'] = documents.index(page) + 1
                except:
                    # Fallback if index can't be determined
                    page.metadata['page'] = 0
           
            # Instead of chunking each page into multiple pieces,
            # treat each page as a single chunk
            chunk = page
            
            # Add file path to metadata if not present
            if 'file_path' not in chunk.metadata:
                chunk.metadata['file_path'] = file_path
           
            all_chunks.append(chunk)
       
        logger.info(f"Created {len(all_chunks)} chunks from {page_count} pages in PDF: {file_path}")
        return all_chunks
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise

def doc_chunking(file_path):
    """Load and chunk a Word document"""
    try:
        loader = Docx2txtLoader(file_path)
        data = loader.load()
        chunks = []
        for i, page in enumerate(data):
            # Add page number if not present
            if 'page' not in page.metadata and 'page_number' not in page.metadata:
                page.metadata['page'] = i + 1
            page_chunks = chunk_page(page)
            chunks.extend(page_chunks)
        logger.info(f"Chunked DOCX: {file_path} into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking DOCX {file_path}: {str(e)}")
        raise

def ppt_chunking(file_path):
    """Load and chunk a PowerPoint file with improved structure preservation"""
    try:
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
        documents = loader.load()
        
        # Group content by slide
        slides = {}
        for doc in documents:
            page_number = doc.metadata.get('page_number', 0)
            if isinstance(page_number, str):
                try:
                    page_number = int(page_number)
                except ValueError:
                    page_number = 0
            
            if page_number not in slides:
                slides[page_number] = {
                    'title': '',
                    'content': []
                }
            
            # Extract titles from heading elements
            element_type = doc.metadata.get('category', '').lower()
            if element_type == 'Title' or element_type == 'Header':
                slides[page_number]['title'] = doc.page_content
            else:
                slides[page_number]['content'].append(doc.page_content)
        
        # Create structured chunks
        all_chunks = []
        for slide_num in sorted(slides.keys()):
            slide = slides[slide_num]
            
            # Create well-formatted slide content
            slide_content = f"Slide {slide_num}\n"
            if slide['title']:
                slide_content += f"Title: {slide['title']}\n\n"
            slide_content += "\n".join(slide['content'])
            
            # Skip empty slides
            if len(slide_content.strip()) <= 10:  # Arbitrary threshold
                continue
                
            chunk = Document(
                page_content=slide_content,
                metadata={
                    "source": file_path, 
                    "page": slide_num,
                    "file_type": "ppt",
                    "title": os.path.basename(file_path)
                }
            )
            
            chunk.metadata = normalize_metadata(chunk.metadata)
            all_chunks.append(chunk)
            
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error in ppt_chunking: {str(e)}")
        return []

def excel_chunking(file_path):
    """Load an Excel file with proper metadata"""
    try:
        # Load the documents
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        docs = loader.load()
        
        # Group by sheet or create a single combined document
        combined_content = ""
        for doc in docs:
            combined_content += doc.page_content + "\n"
        
        # Create a single Document with all content and standardized metadata
        single_chunk = Document(
            page_content=combined_content.strip(),
            metadata={
                "source": file_path,
                "page": 1,  # Set a default page number
                "file_path": file_path
            }
        )
        
        # Normalize metadata
        single_chunk.metadata = normalize_metadata(single_chunk.metadata)
        
        logger.info(f"Loaded Excel: {file_path} as a single chunk")
        return [single_chunk]  # Return a list with just one chunk
    except Exception as e:
        logger.error(f"Error loading Excel {file_path}: {str(e)}")
        raise

def html_chunking(file_path):
    """Load and chunk an HTML file"""
    try:
        loader = UnstructuredHTMLLoader(file_path)
        data = loader.load()
        chunks = []
        for i, page in enumerate(data):
            # Add page number if not present
            if 'page' not in page.metadata and 'page_number' not in page.metadata:
                page.metadata['page'] = i + 1
            page_chunks = chunk_page(page)
            chunks.extend(page_chunks)
        logger.info(f"Chunked HTML: {file_path} into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking HTML {file_path}: {str(e)}")
        raise
    
def text_chunking(file_path):
    """Load and chunk a text file"""
    try:
        loader = TextLoader(file_path)
        data = loader.load()
        chunks = []
        for i, page in enumerate(data):
            # Add page number if not present
            if 'page' not in page.metadata and 'page_number' not in page.metadata:
                page.metadata['page'] = i + 1
            page_chunks = chunk_page(page)
            chunks.extend(page_chunks)
        logger.info(f"Chunked text: {file_path} into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text file {file_path}: {str(e)}")
        raise

def load_and_split_document(file_path: str) -> List[Document]:
    """
    Load and split a document based on file extension
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    if file_extension == '.pdf':
        return pdf_chunking(file_path)
    elif file_extension == '.docx':
        return doc_chunking(file_path)
    elif file_extension in ['.pptx', '.ppt']:
        return ppt_chunking(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return excel_chunking(file_path)
    elif file_extension in ['.html', '.htm']:
        return html_chunking(file_path)
    elif file_extension in ['.txt']:
        return text_chunking(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")