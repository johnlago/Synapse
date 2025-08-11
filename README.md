# Local RAG Database

A containerized RAG (Retrieval-Augmented Generation) system using Chroma, Ollama, and Unstructured for local document processing and querying via MCP (Model Context Protocol).

## Architecture

- **Ollama**: Local embedding model (Qwen2.5-Embedding-4B)
- **Chroma**: Vector database for document storage
- **Unstructured**: Document processing and chunking
- **MCP Server**: Claude Code integration for RAG queries
- **Processor**: File watching and document ingestion

## Quick Start

1. **Install and start Ollama locally:**
   ```bash
   # Install Ollama (if not installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
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

## MCP Integration

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "python",
      "args": ["/path/to/Local-RAG-DB/mcp-server/server.py"],
      "env": {
        "CHROMA_HOST": "localhost:8000",
        "OLLAMA_HOST": "localhost:11434"
      }
    }
  }
}
```

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
- `EMBEDDING_MODEL`: Ollama embedding model (default: qwen2.5-embedding:4b)
- `CHROMA_HOST`: Chroma database host
- `OLLAMA_HOST`: Ollama service host

## File Watching

The system automatically watches the `./documents` folder and processes new/modified files in real-time.