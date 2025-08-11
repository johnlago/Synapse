# Local LLM RAG Chat + MCP

A containerized RAG (Retrieval-Augmented Generation) system using Chroma, Ollama, and Unstructured for local notes/documents processing and querying via MCP (Model Context Protocol) or a dedicated UI.

- Uses Ollama for full local ingestion + RAG datababase.
- Also provides an MCP server to be added to tools like Claude

## Architecture

- **Ollama**: Local embedding model (mxbai-embed-large)
- **Chroma**: Vector database for document storage
- **Unstructured**: Document processing and chunking
- **MCP Server**: Claude Code integration for RAG queries
- **Processor**: File watching and document ingestion

## Quick Start

1. **Install and start Ollama locally:**


Note: It is best to run Ollama natively rather than on a container -- Makes things easier to use GPU

   ```bash
   # Install Ollama (if not installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   
   # Pull the embedding model
   ollama pull mxbai-embed-large
   ```

2. **Start the Docker services:**
   ```bash
   docker-compose up -d
   ```

3. **Add documents:**
   ```bash
   # Copy your documents to the documents folder
   cp /path/to/your/notes/*.md ./documents/
   ```

4. **Process documents:**
   ```bash
   curl -X POST http://localhost:8001/process-all
   ```

5. **Check status:**
   ```bash
   curl http://localhost:8001/status
   ```
6. **Reset Everything --  delete the collection in chroma**
   ```bash
   curl http://localhost:8001/reset
   ```

## Chat Interface with Local LLM

A web-based chat interface that uses your local Ollama installation to answer questions about your documents.

**Access:** http://localhost:8003

**Features:**
- **Smart Document Search**: Automatically searches your notes when you ask questions about topics, projects, or specific information
- **Conversational Interface**: Natural language responses with context from your documents  
- **Real-time Streaming**: Live response streaming for smooth conversation flow
- **Stateless Sessions**: Each message is independent - no conversation history carried between messages
- **Fallback Handling**: Gracefully handles greetings and general chat without unnecessary document searches

**Example Usage:**
- "What did I write about Redis performance?" → Searches documents and provides contextual response 
- "hi" → Normal greeting response without document search
- "What's the status of my collection?" → Shows document indexing information

**Requirements:**
- Ollama running locally with a chat model (default: `llama3.1:8b`)
- Processor service running for document search API
- Documents already indexed in ChromaDB

## MCP Integration with Claude Code

> Note: This is not local anymore -- Claude can now read your notes. Proceed with caution

```bash
claude mcp add local-rag -- docker exec -i local-rag-db-mcp-server-1 python /app/server_fastmcp.py
```

Update the path in the second method to match your actual project location.

## Services & Ports

- **ChromaDB**: Port 8000 - Vector database
- **Processor**: Port 8001 - Document processing and search API  
- **MCP Server**: Port 8002 - Model Context Protocol server
- **Chat Interface**: Port 8003 - Web-based chat UI

## API Endpoints

### Processor Service (Port 8001)
- `POST /process` - Process a specific file
- `POST /process-all` - Process all documents
- `POST /search` - Search documents (used by chat interface)
- `GET /status` - System status
- `POST /reset` - Reset/clear the document collection

### Chat Interface (Port 8003)
- `GET /` - Web chat interface
- `WebSocket /ws/chat` - Real-time chat communication
- `GET /health` - Service health check

### MCP Tools
- `search_documents` - Semantic search across all documents
- `get_document_info` - Document collection statistics
- `search_by_source` - Search within specific files

## Supported Formats

- Markdown (.md)
- PDF (.pdf)  
- Text (.txt)
- Word documents (.docx)

## Configuration

Environment variables:
- `EMBEDDING_MODEL`: Ollama embedding model (default: mxbai-embed-large)
- `CHROMA_HOST`: Chroma database host
- `OLLAMA_HOST`: Ollama service host
- `BATCH_SIZE`: Number of files to process in each batch (default: 10)
- `OLLAMA_RETRIES`: Number of retry attempts for Ollama requests (default: 3)

## Resource Limits

Docker containers have the following resource limits:
- **ChromaDB**: 1GB RAM, 0.5 CPU cores
- **Processor**: 2GB RAM, 1 CPU core  
- **MCP Server**: 1GB RAM, 0.5 CPU cores
- **Chat Interface**: 1GB RAM, 0.5 CPU cores

Adjust these according to your preference

## File Watching

The system automatically watches the `./documents` folder and processes new/modified files in real-time.