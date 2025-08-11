# Local RAG Database

A containerized RAG (Retrieval-Augmented Generation) system using Chroma, Ollama, and Unstructured for local document processing and querying via MCP (Model Context Protocol).

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

## MCP Integration with Claude Code

```bash
claude mcp add local-rag -- docker exec -i local-rag-db-mcp-server-1 python /app/server_fastmcp.py
```

Update the path in the second method to match your actual project location.

## API Endpoints

### Processor Service (Port 8001)
- `POST /process` - Process a specific file
- `POST /process-all` - Process all documents
- `GET /status` - System status

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

## File Watching

The system automatically watches the `./documents` folder and processes new/modified files in real-time.