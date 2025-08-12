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

## Linux Networking Issues and Workarounds

### Ollama IPv6/IPv4 Binding Issue

On Linux systems, Ollama has a known networking issue where it doesn't properly bind to both IPv4 and IPv6:

- **Setting `OLLAMA_HOST=0.0.0.0`**: Makes Ollama listen only on IPv6, breaking IPv4 connectivity
- **Default configuration**: Makes Ollama listen only on loopback (127.0.0.1), preventing Docker container access

### Recommended Solution: Host Networking (Used by this project)

This project uses **host networking** for services that need Ollama access as the most reliable workaround:

**Advantages:**
- ✅ Bypasses Docker's network isolation issues with Ollama
- ✅ Works consistently across different Linux distributions
- ✅ No need to modify Ollama's systemd configuration
- ✅ Automatic OS detection via Makefile

**Security Considerations:**
- ⚠️ Services using host networking expose ports directly on your host
- ⚠️ Potential port conflicts with other services
- ⚠️ Reduced network isolation compared to bridge networking

**Ports exposed on host when using Linux configuration:**
- `8001` - Processor service
- `8002` - MCP server  
- `8003` - Chat interface
- ChromaDB remains isolated in bridge network

### Alternative: Manual Ollama Configuration (Not recommended)

If you prefer to modify Ollama instead of using host networking:

1. Edit Ollama service: `sudo vim /usr/lib/systemd/system/ollama.service`
2. Add: `Environment="OLLAMA_HOST=0.0.0.0"`
3. Reload and restart:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

**⚠️ Security Warning**: This exposes Ollama to your entire local network, allowing anyone on your network to access your Ollama instance if your firewall permits.

## Quick Start

1. **Install and start Ollama locally:**

Check [HERE](https://ollama.com/library) and find a model that will work on your machine.
The higher the parameter count, the better it is, and heavier it will be on your CPU/GPU.

If unsure, start with `llama3.1:8b`

Note: It is best to run Ollama natively rather than on a container -- Makes things easier to use GPU

   ```bash
   # Install Ollama (if not installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   
   # Pull the embedding model
   ollama pull mxbai-embed-large

   # pull the chat model
   ollama pull llama3.1:8b
   ```

2. **Configure document path (optional):**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env to point to your documents directory
   # Example: DOCUMENTS_PATH=/Users/username/Documents/Notes
   ```

3. **Start the Docker services:**
   ```bash
   # Recommended: Use Makefile (automatically detects OS)
   make up
   
   # Or manually:
   # On macOS: docker-compose up -d
   # On Linux: docker-compose -f docker-compose.yml -f docker-compose.linux.yml up -d
   ```

4. **Process documents:**
   ```bash
   # Using Makefile
   make process-all
   
   # Or directly
   curl -X POST http://localhost:8001/process-all
   ```

5. **Check status:**
   ```bash
   # Using Makefile
   make status
   
   # Or directly
   curl http://localhost:8001/status
   ```

6. **Reset Everything -- delete the collection in chroma:**
   ```bash
   # Using Makefile
   make reset
   
   # Or directly
   curl -X POST http://localhost:8001/reset
   ```

## Makefile Commands

This project includes a Makefile that automatically detects your operating system and uses the appropriate Docker Compose configuration.

### Available Commands

```bash
make help           # Show all available commands
make up             # Start all services (OS-aware)
make down           # Stop all services
make logs           # Show logs for all services
make logs-processor # Show logs for processor service only
make logs-chat      # Show logs for chat interface only
make logs-mcp       # Show logs for MCP server only
make build          # Build all services
make rebuild        # Rebuild and restart all services
make status         # Check system status
make process-all    # Process all documents
make reset          # Reset document collection
make search QUERY="search terms"  # Search documents
make clean          # Stop and remove all containers and volumes
```

### OS Detection

The Makefile automatically detects your operating system:
- **Linux**: Uses `docker-compose -f docker-compose.yml -f docker-compose.linux.yml`
- **macOS**: Uses standard `docker-compose`

Run `make help` to see which configuration is being used on your system.

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

Environment variables (configure via `.env` file):
- `DOCUMENTS_PATH`: Absolute path to your documents directory (default: ./documents)
- `EMBEDDING_MODEL`: Ollama embedding model (default: mxbai-embed-large)
- `CHAT_MODEL`: Ollama chat model for the interface (default: llama3.1:8b)
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