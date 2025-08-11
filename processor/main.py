import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Dict
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
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "3"))

logger.info(f"Configuration loaded:")
logger.info(f"  OLLAMA_HOST: {OLLAMA_HOST}")
logger.info(f"  CHROMA_HOST: {CHROMA_HOST}")
logger.info(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"  DOCUMENTS_PATH: {DOCUMENTS_PATH}")
logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
logger.info(f"  OLLAMA_RETRIES: {OLLAMA_RETRIES}")

# Initialize clients
logger.info("Initializing Chroma client...")
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST.split(':')[0], port=int(CHROMA_HOST.split(':')[1]))
    logger.info("Chroma client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Chroma client: {e}")
    chroma_client = None

collection = None

# Simple lock to prevent concurrent processing of same file
processing_files = set()
processing_lock = asyncio.Lock()

class ProcessRequest(BaseModel):
    file_path: str
    force_reprocess: bool = False

class FileWatcher(FileSystemEventHandler):
    def __init__(self, processor_func, loop):
        self.processor_func = processor_func
        self.loop = loop
        
    def _schedule_processing(self, file_path, event_type):
        """Schedule async processing from thread"""
        try:
            asyncio.run_coroutine_threadsafe(
                self.processor_func(file_path), 
                self.loop
            )
        except Exception as e:
            logger.error(f"Error scheduling processing for {file_path}: {e}")
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.md', '.pdf', '.txt', '.docx')):
            logger.info(f"File created: {event.src_path}")
            self._schedule_processing(event.src_path, "created")
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.md', '.pdf', '.txt', '.docx')):
            logger.info(f"File modified: {event.src_path}")
            self._schedule_processing(event.src_path, "modified")

async def get_embeddings(text: str) -> List[float]:
    """Get embeddings from Ollama with retry logic"""
    import ollama
    client = ollama.Client(host=f"http://{OLLAMA_HOST}")
    
    for attempt in range(OLLAMA_RETRIES):
        try:
            response = client.embeddings(
                model=EMBEDDING_MODEL,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            if attempt == OLLAMA_RETRIES - 1:  # Last attempt
                logger.error(f"Error getting embeddings from {OLLAMA_HOST} after {OLLAMA_RETRIES} attempts: {e}")
                raise
            else:
                logger.warning(f"Embedding attempt {attempt + 1} failed, retrying: {e}")
                await asyncio.sleep(2)  # Simple 2-second delay

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
        # Prevent concurrent processing of the same file
        async with processing_lock:
            if file_path in processing_files:
                logger.info(f"Document {file_path} already being processed, skipping")
                return
            
            # Check if already processed (unless force reprocess)
            if not force_reprocess:
                existing = collection.get(where={"source": file_path})
                if existing['ids']:
                    logger.info(f"Document {file_path} already processed, skipping")
                    return
            
            # Mark as being processed
            processing_files.add(file_path)
        
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
        
        # Add to Chroma - use relative path from documents root to avoid collisions
        relative_path = os.path.relpath(file_path, DOCUMENTS_PATH)
        # Replace path separators with underscores for valid ID
        safe_path = relative_path.replace(os.sep, '_').replace('.', '_')
        ids = [f"{safe_path}_{chunk['metadata']['chunk_id']}" for chunk in chunks]
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
    finally:
        # Always remove from processing set
        processing_files.discard(file_path)

@app.on_event("startup")
async def startup():
    global collection
    
    logger.info("=== STARTUP SEQUENCE BEGINNING ===")
    
    # Wait for Ollama to be ready and pull model - FAIL FAST
    logger.info("Waiting for Ollama to be ready...")
    ollama_attempts = 0
    ollama_client = None
    max_ollama_attempts = 12  # 1 minute timeout
    
    while ollama_attempts < max_ollama_attempts:
        try:
            logger.info(f"Attempting to connect to Ollama at {OLLAMA_HOST} (attempt {ollama_attempts + 1})")
            ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}")
            ollama_client.list()
            logger.info("✅ Ollama connection successful!")
            break
        except Exception as e:
            ollama_attempts += 1
            logger.warning(f"❌ Ollama connection failed (attempt {ollama_attempts}): {e}")
            if ollama_attempts >= max_ollama_attempts:
                logger.error("🚨 CRITICAL: Failed to connect to Ollama after 1 minute!")
                logger.error("🚨 Check if Ollama is running and accessible at: " + OLLAMA_HOST)
                logger.error("🚨 Container will exit now - this is intentional!")
                exit(1)
            await asyncio.sleep(5)
    
    if ollama_client is None:
        logger.error("🚨 CRITICAL: Ollama client is None - this should not happen!")
        exit(1)
    
    # Pull embedding model if not exists
    if ollama_client:
        try:
            logger.info("Checking if embedding model exists...")
            response = ollama_client.list()
            models = response.get('models', [])
            model_names = [model.get('name', model.get('model', 'unknown')) for model in models]
            logger.info(f"Available models: {model_names}")
            
            # Check if our embedding model exists (with or without :latest tag)
            model_found = any(
                EMBEDDING_MODEL in name or f"{EMBEDDING_MODEL}:latest" in name 
                for name in model_names
            )
            
            if not model_found:
                logger.info(f"📥 Pulling embedding model: {EMBEDDING_MODEL}")
                ollama_client.pull(EMBEDDING_MODEL)
                logger.info(f"✅ Successfully pulled {EMBEDDING_MODEL}")
            else:
                logger.info(f"✅ Model {EMBEDDING_MODEL} already available")
        except Exception as e:
            logger.error(f"❌ Error with Ollama model setup: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    else:
        logger.warning("⚠️ Ollama client not available, skipping model check")
    
    # Initialize Chroma collection - FAIL FAST
    try:
        logger.info("Connecting to Chroma database...")
        if chroma_client is None:
            logger.error("🚨 CRITICAL: Chroma client is None!")
            logger.error("🚨 Check if ChromaDB is running and accessible at: " + CHROMA_HOST)
            logger.error("🚨 Container will exit now - this is intentional!")
            exit(1)
            
        # Get or create collection with cosine similarity
        try:
            collection = chroma_client.get_collection("documents")
            logger.info("✅ Found existing collection")
        except:
            logger.info("Creating new collection...")
            collection = chroma_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ Created new collection")
        logger.info("✅ Connected to Chroma collection successfully")
    except Exception as e:
        logger.error(f"🚨 CRITICAL: Error connecting to Chroma: {e}")
        logger.error("🚨 Check if ChromaDB is running and accessible at: " + CHROMA_HOST)
        logger.error("🚨 Container will exit now - this is intentional!")
        exit(1)
    
    # Start file watcher
    try:
        logger.info(f"Starting file watcher for {DOCUMENTS_PATH}...")
        loop = asyncio.get_running_loop()
        event_handler = FileWatcher(process_document, loop)
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

async def process_batch(file_paths: List[str], batch_num: int):
    """Process a batch of files"""
    logger.info(f"Processing batch {batch_num} with {len(file_paths)} files")
    for file_path in file_paths:
        try:
            await process_document(file_path)
        except Exception as e:
            logger.error(f"Failed to process {file_path} in batch {batch_num}: {e}")
    logger.info(f"Completed batch {batch_num}")

@app.post("/process-all")
async def process_all_documents(background_tasks: BackgroundTasks):
    """Process all documents in the documents folder in batches"""
    # Collect all files first
    all_files = []
    for root, dirs, files in os.walk(DOCUMENTS_PATH):
        for file in files:
            if file.endswith(('.md', '.pdf', '.txt', '.docx')):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    # Process in batches
    total_files = len(all_files)
    total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    logger.info(f"Found {total_files} files, processing in {total_batches} batches of {BATCH_SIZE}")
    
    for i in range(0, total_files, BATCH_SIZE):
        batch = all_files[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        background_tasks.add_task(process_batch, batch, batch_num)
    
    return {
        "message": f"Processing {total_files} documents in {total_batches} batches",
        "batch_size": BATCH_SIZE,
        "total_batches": total_batches
    }

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search documents using semantic similarity"""
    try:
        query = request.query
        limit = request.limit
        
        if not query:
            return {"error": "Query is required"}
        
        if not collection:
            return {"error": "No collection available"}
        
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Search in Chroma using cosine similarity
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return {"results": [], "message": "No documents found"}
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Debug: Log raw distance values
            logger.info(f"Raw distance: {distance}, type: {type(distance)}")
            
            # With cosine similarity, distance is 1 - cosine_similarity, so similarity = 1 - distance
            similarity = max(0, 1 - distance)  # Convert cosine distance to similarity [0,1]
            logger.info(f"Calculated similarity: {similarity}")
            
            formatted_results.append({
                "content": doc,
                "similarity": round(similarity, 3),
                "source": metadata.get('file_name', 'Unknown'),
                "chunk_id": metadata.get('chunk_id', i)
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": str(e)}

@app.post("/reset")
async def reset_database():
    """Reset the ChromaDB by deleting and recreating the collection"""
    global collection
    try:
        if chroma_client is None:
            return {"error": "Chroma client not available"}
        
        # Delete existing collection
        try:
            chroma_client.delete_collection("documents")
            logger.info("✅ Deleted existing collection")
        except Exception as e:
            logger.warning(f"Could not delete collection (may not exist): {e}")
        
        # Recreate collection
        collection = chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("✅ Created new collection")
        
        return {
            "status": "success",
            "message": "Database reset successfully",
            "documents_indexed": 0
        }
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return {"error": str(e)}

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