# Synapse: Local RAG System with MCP Integration

A containerized RAG (Retrieval-Augmented Generation) system that indexes your local documents and makes them searchable through Claude Code and LM Studio via MCP (Model Context Protocol).


> Note: This is primarily tested on Mac and Linux (check LINUX.md)
> No clue about Windows.

**Key Features:**
- 🔒 **Fully Local**: Uses Ollama for embeddings and LLM processing
- 🔌 **MCP Integration**: Works seamlessly with Claude Code and LM Studio
- 📁 **Multi-Format Support**: Processes Markdown, PDF, Word docs, and text files
- ⚡ **Real-Time Processing**: Auto-detects and processes new/modified documents
- 🌐 **Web Interface**: Built-in chat interface for direct document querying (very basic, better to use LMStudio)

> **📋 Documentation Links:**
> - [Architecture Overview](ARCHITECTURE.md) - Detailed system architecture with diagrams
> - [Linux Configuration](LINUX.md) - Linux-specific setup and troubleshooting

## Quick Setup

### 1. Install Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull mxbai-embed-large  # Embedding model
ollama pull llama3.1:8b        # Chat model
```

### 2. Configure Documents Path
```bash
cp .env.example .env
# Edit .env to set DOCUMENTS_PATH to your documents directory
```

### 3. Start Services
```bash
# Automatically detects your OS (Linux/macOS)
make up

# Process your documents
make process-all

# Check system status
make status
```

### 4. Verify Setup
- **Web Interface**: http://localhost:8003
- **API Status**: http://localhost:8001/status

## Using with LM Studio

[LM Studio](https://lmstudio.ai/) provides an excellent local LLM interface that can be enhanced with your document search capabilities via MCP.

### Prerequisites
- LM Studio version 0.3.17 or later
- This RAG system running (`make up`)

### Setup MCP Integration

1. **Open LM Studio** and navigate to the **Program** tab
2. **Edit mcp.json** and add the synapse server:

```json
{
  "mcpServers": {
    "synapse": {
      "command": "docker",
      "args": [
        "exec", "-i", 
        "synapse-mcp-server-1", 
        "python", "/app/server_fastmcp.py"
      ],
      "env": {}
    }
  }
}
```

3. **Restart LM Studio** to load the MCP server
4. **Test the integration** by asking questions about your documents

### Available MCP Tools
- `search_documents` - Semantic search across all your documents
- `get_document_info` - Get statistics about your document collection
- `search_by_source` - Search within specific files

## Using with Claude Code

> ⚠️ **Privacy Note**: This enables Claude to access your local documents. Use with caution.

### Setup
```bash
# Add the MCP server to Claude Code
claude mcp add synapse -- docker exec -i synapse-mcp-server-1 python /app/server_fastmcp.py
```

### Usage
Once configured, Claude Code can automatically search and reference your local documents when answering questions:

- "What did I write about database optimization?"
- "Find my notes on React performance patterns"
- "Show me documentation about the authentication system"

## Built-in Web Interface

For direct local usage without external tools:

**Access:** http://localhost:8003

**Features:**
- Real-time streaming responses
- Automatic document search
- Stateless conversations
- Fallback for general questions

## Makefile Commands

```bash
make help           # Show all available commands
make up             # Start all services (auto-detects OS)
make down           # Stop all services
make status         # Check system status
make process-all    # Process all documents
make reset          # Reset document collection
make logs           # Show logs for all services
make clean          # Remove all containers and volumes
```

> **💡 Tip**: Run `make help` to see OS-specific configuration being used.

## Configuration

### Environment Variables (.env file)
- `DOCUMENTS_PATH`: Path to your documents directory
- `EMBEDDING_MODEL`: Ollama embedding model (default: mxbai-embed-large)
- `CHAT_MODEL`: Ollama chat model (default: llama3.1:8b)

### Supported Document Formats
- Markdown (.md), PDF (.pdf), Text (.txt), Word (.docx)

### Services & Ports
- **ChromaDB**: :8000 (Vector database)
- **Processor**: :8001 (Document processing & search)
- **MCP Server**: :8002 (Model Context Protocol)
- **Chat Interface**: :8003 (Web UI)

## Troubleshooting

- **Linux users**: See [LINUX.md](LINUX.md) for networking configuration
- **System logs**: `make logs` or `make logs-processor`
- **Reset everything**: `make reset && make process-all`
- **Clean install**: `make clean && make up`

