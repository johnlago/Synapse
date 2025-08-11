#!/usr/bin/env python3
import os
import asyncio
import logging
from typing import Any, List, Dict
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import chromadb
import ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-mcp-server")

# Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost:8000")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434") 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

# Initialize clients
chroma_client = None
collection = None

app = Server("local-rag")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available RAG tools"""
    return [
        Tool(
            name="search_documents",
            description="Search through indexed documents using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer", 
                        "description": "Maximum number of results to return",
                        "default": 5
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0 to 1.0)",
                        "default": 0.3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_document_info",
            description="Get information about indexed documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Optional: Get info for specific file"
                    }
                }
            }
        ),
        Tool(
            name="search_by_source",
            description="Search within specific document sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "source_filter": {
                        "type": "string",
                        "description": "Filter by source file name or path"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query", "source_filter"]
            }
        )
    ]

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

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    global chroma_client, collection
    
    # Initialize clients if needed
    if not chroma_client:
        try:
            chroma_client = chromadb.HttpClient(
                host=CHROMA_HOST.split(':')[0], 
                port=int(CHROMA_HOST.split(':')[1])
            )
            collection = chroma_client.get_collection("documents")
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error connecting to Chroma database: {e}"
            )]
    
    if name == "search_documents":
        try:
            query = arguments["query"]
            limit = arguments.get("limit", 5)
            min_similarity = arguments.get("min_similarity", 0.3)
            
            # Get query embedding
            query_embedding = await get_embeddings(query)
            
            # Search in Chroma
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'][0]:
                return [TextContent(
                    type="text",
                    text="No documents found matching your query."
                )]
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                similarity = 1 - distance  # Convert distance to similarity
                if similarity >= min_similarity:
                    formatted_results.append(
                        f"**Result {i+1}** (similarity: {similarity:.3f})\n"
                        f"**Source:** {metadata.get('file_name', 'Unknown')}\n"
                        f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\n"
                    )
            
            if not formatted_results:
                return [TextContent(
                    type="text", 
                    text=f"No documents found with similarity >= {min_similarity}"
                )]
            
            return [TextContent(
                type="text",
                text=f"Found {len(formatted_results)} relevant documents:\n\n" + 
                     "\n---\n".join(formatted_results)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error searching documents: {e}"
            )]
    
    elif name == "get_document_info":
        try:
            file_name = arguments.get("file_name")
            
            if file_name:
                # Get info for specific file
                results = collection.get(
                    where={"file_name": file_name},
                    include=["metadatas"]
                )
                if not results['ids']:
                    return [TextContent(
                        type="text",
                        text=f"No document found with name: {file_name}"
                    )]
                
                return [TextContent(
                    type="text",
                    text=f"Document: {file_name}\n"
                         f"Chunks: {len(results['ids'])}\n"
                         f"Last processed: {results['metadatas'][0].get('processed_at', 'Unknown')}"
                )]
            else:
                # Get overall collection info
                count = collection.count()
                
                # Get unique files
                all_metadata = collection.get(include=["metadatas"])
                unique_files = set()
                for metadata in all_metadata['metadatas']:
                    unique_files.add(metadata.get('file_name', 'Unknown'))
                
                return [TextContent(
                    type="text",
                    text=f"RAG Database Status:\n"
                         f"Total chunks: {count}\n"
                         f"Unique documents: {len(unique_files)}\n"
                         f"Documents: {', '.join(sorted(unique_files))}"
                )]
                
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error getting document info: {e}"
            )]
    
    elif name == "search_by_source":
        try:
            query = arguments["query"]
            source_filter = arguments["source_filter"]
            limit = arguments.get("limit", 5)
            
            # Get query embedding
            query_embedding = await get_embeddings(query)
            
            # Search with source filter
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get more to filter
                where={"file_name": {"$contains": source_filter}},
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'][0]:
                return [TextContent(
                    type="text",
                    text=f"No documents found in source containing '{source_filter}'"
                )]
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0][:limit],
                results['metadatas'][0][:limit],
                results['distances'][0][:limit]
            )):
                similarity = 1 - distance
                formatted_results.append(
                    f"**Result {i+1}** (similarity: {similarity:.3f})\n"
                    f"**Source:** {metadata.get('file_name', 'Unknown')}\n" 
                    f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\n"
                )
            
            return [TextContent(
                type="text",
                text=f"Found {len(formatted_results)} results in '{source_filter}':\n\n" +
                     "\n---\n".join(formatted_results)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error searching by source: {e}"
            )]
    
    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    """Run the MCP server"""
    logger.info("Starting Local RAG MCP Server")
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())