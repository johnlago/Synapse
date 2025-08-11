import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import chromadb
import ollama
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local RAG Processor")

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost:8000")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
DOCUMENTS_PATH = "/app/documents"

# Initialize clients
chroma_client = chromadb.HttpClient(host=CHROMA_HOST.split(':')[0], port=int(CHROMA_HOST.split(':')[1]))
collection = None

class ProcessRequest(BaseModel):
    file_path: str
    force_reprocess: bool = False

class FileWatcher(FileSystemEventHandler):
    def __init__(self, processor_func):
        self.processor_func = processor_func
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.md', '.pdf', '.txt', '.docx')):
            logger.info(f"File modified: {event.src_path}")
            asyncio.create_task(self.processor_func(event.src_path))

async def get_embeddings(text: str) -> List[float]:
    """Get embeddings from Ollama"""
    try:
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise

def chunk_document(file_path: str) -> List[Dict[str, Any]]:
    """Process and chunk a document using unstructured"""
    try:
        # Partition the document
        elements = partition(filename=file_path)
        
        # Chunk by title for better semantic coherence
        chunks = chunk_by_title(
            elements,
            max_characters=1000,
            combine_text_under_n_chars=100,
            new_after_n_chars=800
        )
        
        # Convert to our format
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "text": str(chunk),
                "metadata": {
                    "source": file_path,
                    "chunk_id": i,
                    "element_type": getattr(chunk, 'category', 'text'),
                    "file_name": Path(file_path).name,
                    "processed_at": time.time()
                }
            })
        
        return processed_chunks
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

async def process_document(file_path: str, force_reprocess: bool = False):
    """Process a single document and add to vector database"""
    try:
        # Check if already processed (unless force reprocess)
        if not force_reprocess:
            existing = collection.get(where={"source": file_path})
            if existing['ids']:
                logger.info(f"Document {file_path} already processed, skipping")
                return
        
        logger.info(f"Processing document: {file_path}")
        
        # Chunk the document
        chunks = chunk_document(file_path)
        if not chunks:
            logger.warning(f"No chunks extracted from {file_path}")
            return
        
        # Get embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            embedding = await get_embeddings(chunk["text"])
            embeddings.append(embedding)
        
        # Add to Chroma
        ids = [f"{Path(file_path).stem}_{chunk['metadata']['chunk_id']}" for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully processed {len(chunks)} chunks from {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")

@app.on_event("startup")
async def startup():
    global collection
    
    # Wait for Ollama to be ready and pull model
    logger.info("Waiting for Ollama...")
    while True:
        try:
            ollama.list()
            break
        except:
            await asyncio.sleep(5)
    
    # Pull embedding model if not exists
    try:
        models = ollama.list()['models']
        if not any(EMBEDDING_MODEL in model['name'] for model in models):
            logger.info(f"Pulling embedding model: {EMBEDDING_MODEL}")
            ollama.pull(EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Error with Ollama setup: {e}")
    
    # Initialize Chroma collection
    try:
        collection = chroma_client.get_or_create_collection(
            name="documents",
            metadata={"description": "Local RAG document collection"}
        )
        logger.info("Connected to Chroma collection")
    except Exception as e:
        logger.error(f"Error connecting to Chroma: {e}")
    
    # Start file watcher
    event_handler = FileWatcher(process_document)
    observer = Observer()
    observer.schedule(event_handler, DOCUMENTS_PATH, recursive=True)
    observer.start()
    logger.info(f"Started watching {DOCUMENTS_PATH}")

@app.post("/process")
async def process_file(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Process a specific file"""
    file_path = os.path.join(DOCUMENTS_PATH, request.file_path)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    background_tasks.add_task(process_document, file_path, request.force_reprocess)
    return {"message": f"Processing {request.file_path}"}

@app.post("/process-all")
async def process_all_documents(background_tasks: BackgroundTasks):
    """Process all documents in the documents folder"""
    processed = 0
    for root, dirs, files in os.walk(DOCUMENTS_PATH):
        for file in files:
            if file.endswith(('.md', '.pdf', '.txt', '.docx')):
                file_path = os.path.join(root, file)
                background_tasks.add_task(process_document, file_path)
                processed += 1
    
    return {"message": f"Processing {processed} documents"}

@app.get("/status")
async def get_status():
    """Get system status"""
    try:
        doc_count = collection.count() if collection else 0
        return {
            "status": "healthy",
            "documents_indexed": doc_count,
            "embedding_model": EMBEDDING_MODEL,
            "chroma_host": CHROMA_HOST
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)