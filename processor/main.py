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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting Local RAG Processor...")

app = FastAPI(title="Local RAG Processor")

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "host.docker.internal:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma:8000")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
DOCUMENTS_PATH = "/app/documents"

logger.info(f"Configuration loaded:")
logger.info(f"  OLLAMA_HOST: {OLLAMA_HOST}")
logger.info(f"  CHROMA_HOST: {CHROMA_HOST}")
logger.info(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"  DOCUMENTS_PATH: {DOCUMENTS_PATH}")

# Initialize clients
logger.info("Initializing Chroma client...")
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST.split(':')[0], port=int(CHROMA_HOST.split(':')[1]))
    logger.info("Chroma client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Chroma client: {e}")
    chroma_client = None

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
        # Configure Ollama client with custom host
        import ollama
        client = ollama.Client(host=f"http://{OLLAMA_HOST}")
        
        response = client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Error getting embeddings from {OLLAMA_HOST}: {e}")
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
    
    logger.info("=== STARTUP SEQUENCE BEGINNING ===")
    
    # Wait for Ollama to be ready and pull model
    logger.info("Waiting for Ollama to be ready...")
    ollama_attempts = 0
    ollama_client = None
    
    while True:
        try:
            logger.info(f"Attempting to connect to Ollama at {OLLAMA_HOST} (attempt {ollama_attempts + 1})")
            ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}")
            ollama_client.list()
            logger.info("✅ Ollama connection successful!")
            break
        except Exception as e:
            ollama_attempts += 1
            logger.warning(f"❌ Ollama connection failed (attempt {ollama_attempts}): {e}")
            if ollama_attempts >= 12:  # 1 minute timeout
                logger.error("❌ Failed to connect to Ollama after 1 minute, continuing anyway...")
                break
            await asyncio.sleep(5)
    
    # Pull embedding model if not exists
    if ollama_client:
        try:
            logger.info("Checking if embedding model exists...")
            models = ollama_client.list()['models']
            model_names = [model['name'] for model in models]
            logger.info(f"Available models: {model_names}")
            
            if not any(EMBEDDING_MODEL in model['name'] for model in models):
                logger.info(f"📥 Pulling embedding model: {EMBEDDING_MODEL}")
                ollama_client.pull(EMBEDDING_MODEL)
                logger.info(f"✅ Successfully pulled {EMBEDDING_MODEL}")
            else:
                logger.info(f"✅ Model {EMBEDDING_MODEL} already available")
        except Exception as e:
            logger.error(f"❌ Error with Ollama model setup: {e}")
    else:
        logger.warning("⚠️ Ollama client not available, skipping model check")
    
    # Initialize Chroma collection
    try:
        logger.info("Connecting to Chroma database...")
        if chroma_client is None:
            logger.error("❌ Chroma client is None, cannot create collection")
            return
            
        collection = chroma_client.get_or_create_collection(
            name="documents",
            metadata={"description": "Local RAG document collection"}
        )
        logger.info("✅ Connected to Chroma collection successfully")
    except Exception as e:
        logger.error(f"❌ Error connecting to Chroma: {e}")
        collection = None
    
    # Start file watcher
    try:
        logger.info(f"Starting file watcher for {DOCUMENTS_PATH}...")
        event_handler = FileWatcher(process_document)
        observer = Observer()
        observer.schedule(event_handler, DOCUMENTS_PATH, recursive=True)
        observer.start()
        logger.info(f"✅ File watcher started for {DOCUMENTS_PATH}")
    except Exception as e:
        logger.error(f"❌ Error starting file watcher: {e}")
    
    logger.info("=== STARTUP SEQUENCE COMPLETE ===")
    logger.info("🚀 Local RAG Processor is ready!")

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
    try:
        logger.info("🔥 Starting uvicorn server...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    except Exception as e:
        logger.error(f"❌ Failed to start uvicorn server: {e}")
        raise