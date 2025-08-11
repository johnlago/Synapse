# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a containerized RAG (Retrieval-Augmented Generation) system that provides local document processing and querying capabilities. The system uses:
- **Ollama** for local embeddings (mxbai-embed-large) and chat models (llama3.1:8b)
- **ChromaDB** as the vector database
- **Unstructured** for document processing and chunking
- **FastAPI** for API services
- **Docker Compose** for orchestration

## Architecture

The system consists of 4 containerized services:
- **ChromaDB** (port 8000): Vector database storage
- **Processor** (port 8001): Document processing, ingestion, and search API
- **MCP Server** (port 8002): Model Context Protocol server for Claude integration
- **Chat Interface** (port 8003): Web-based chat UI with streaming responses

## Development Commands

### Prerequisites
Ollama must be installed and running locally (not containerized for GPU access):
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull mxbai-embed-large
ollama pull llama3.1:8b
```

### Environment Setup
```bash
# Copy and configure environment
cp .env.example .env
# Edit DOCUMENTS_PATH to point to your documents directory
```

### Container Management
```bash
# Start all services
docker-compose up -d

# View logs for specific service
docker-compose logs -f processor
docker-compose logs -f chat-interface

# Rebuild specific service
docker-compose build processor
docker-compose up -d processor

# Stop all services
docker-compose down
```

### Document Processing
```bash
# Process all documents
curl -X POST http://localhost:8001/process-all

# Process specific file
curl -X POST http://localhost:8001/process -H "Content-Type: application/json" -d '{"file_path": "path/to/file.md"}'

# Check system status
curl http://localhost:8001/status

# Reset/clear document collection
curl -X POST http://localhost:8001/reset
```

### Search and Testing
```bash
# Search documents via API
curl -X POST http://localhost:8001/search -H "Content-Type: application/json" -d '{"query": "your search query", "limit": 5}'

# Test chat interface
open http://localhost:8003
```

## Key Configuration

### Environment Variables (via .env file)
- `DOCUMENTS_PATH`: Absolute path to documents directory
- `EMBEDDING_MODEL`: Ollama embedding model (default: mxbai-embed-large)
- `CHAT_MODEL`: Ollama chat model (default: llama3.1:8b)
- `OLLAMA_HOST`: Ollama service host (default: host.docker.internal:11434)
- `BATCH_SIZE`: Files per processing batch (default: 10)
- `OLLAMA_RETRIES`: Retry attempts for Ollama requests (default: 3)

### Supported Document Formats
- Markdown (.md)
- PDF (.pdf)
- Text (.txt)
- Word documents (.docx)

## MCP Integration with Claude Code

Add this system as an MCP server to Claude Code:
```bash
claude mcp add local-rag -- docker exec -i local-rag-db-mcp-server-1 python /app/server_fastmcp.py
```

**Security Note**: MCP integration allows Claude to access your documents remotely.

## File Structure

- `processor/`: Document processing service with FastAPI endpoints, file watching, and Chroma integration
- `mcp-server/`: MCP server implementation for Claude integration
- `chat-interface/`: Web-based chat UI with WebSocket support and streaming responses
- `documents/`: Default document storage (configurable via DOCUMENTS_PATH)

## Common Issues

### Ollama Connection
- Ensure Ollama is running locally: `ollama serve`
- Verify models are pulled: `ollama list`
- Check Docker host connectivity to host.docker.internal

### Document Processing
- File watching monitors `./documents` for real-time processing
- Large documents are chunked automatically using title-based chunking
- Processing uses file checksums to detect changes and avoid reprocessing

### Resource Limits
Services have Docker resource limits:
- ChromaDB: 1GB RAM, 0.5 CPU
- Processor: 2GB RAM, 1.0 CPU  
- MCP Server: 1GB RAM, 0.5 CPU
- Chat Interface: 1GB RAM, 0.5 CPU