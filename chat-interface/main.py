#!/usr/bin/env python3
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-interface")

app = FastAPI(title="Local RAG Chat Interface")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "host.docker.internal:11434")
PROCESSOR_HOST = os.getenv("PROCESSOR_HOST", "processor:8001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")

# Function to detect if a model supports reasoning
def is_reasoning_model(model_name: str) -> bool:
    """Check if the model supports reasoning/thinking tags"""
    reasoning_models = [
        "llama3.2", "qwen", "deepseek", "r1", "thinking", "reasoning",
        "claude", "o1", "marco-o1"
    ]
    return any(reasoning_keyword in model_name.lower() for reasoning_keyword in reasoning_models)

logger.info(f"Configuration:")
logger.info(f"  OLLAMA_HOST: {OLLAMA_HOST}")
logger.info(f"  PROCESSOR_HOST: {PROCESSOR_HOST}")
logger.info(f"  CHAT_MODEL: {CHAT_MODEL}")

# Tool definitions for Ollama
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search through your personal notes and documents using semantic similarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "get_document_status",
            "description": "Get information about the document collection and indexing status",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

class ToolCaller:
    def __init__(self):
        self.http_client = httpx.AsyncClient()
    
    async def search_documents(self, query: str, limit: int = 2) -> str:
        """Search documents via processor API with simple formatting for local LLMs"""
        try:
            url = f"http://{PROCESSOR_HOST}/search"
            payload = {"query": query, "limit": min(limit, 2)}  # Max 2 results for local LLMs
            
            response = await self.http_client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if "error" in data:
                return f"Search error: {data['error']}"
            
            results = data.get("results", [])
            if not results:
                return "No results found."
            
            # Simple format for local LLMs
            formatted_parts = []
            for i, result in enumerate(results, 1):
                source = result.get('source', 'Unknown')
                content = result.get('content', '')[:120] + "..." if len(result.get('content', '')) > 120 else result.get('content', '')
                formatted_parts.append(f"{i}. {source}: {content}")
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return f"Error: {str(e)}"
    
    async def get_document_status(self) -> str:
        """Get status from processor API"""
        try:
            url = f"http://{PROCESSOR_HOST}/status"
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            return (
                f"📊 Document Collection Status:\n"
                f"Documents indexed: {data.get('documents_indexed', 0)}\n"
                f"Embedding model: {data.get('embedding_model', 'Unknown')}\n"
                f"Status: {data.get('status', 'Unknown')}"
            )
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return f"Error getting status: {str(e)}"

tool_caller = ToolCaller()

async def call_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool call"""
    logger.info(f"Calling tool: {name} with args: {arguments}")
    
    if name == "search_documents":
        return await tool_caller.search_documents(
            query=arguments.get("query", ""),
            limit=arguments.get("limit", 5)
        )
    elif name == "get_document_status":
        return await tool_caller.get_document_status()
    else:
        return f"Unknown tool: {name}"

async def chat_with_ollama(message: str) -> AsyncGenerator[str, None]:
    """Stream chat response from Ollama with tool calling using HTTP API"""
    try:
        async with httpx.AsyncClient() as client:
            
            # Check if we should use tools based on keywords
            should_search = any(keyword in message.lower() for keyword in [
                'search', 'find', 'notes', 'documents', 'wrote', 'about', 
                'project', 'status', 'what',
                'show', 'tell', 'explain', 'how', 'when', 'where'
            ])
            
            if should_search and not any(greeting in message.lower() for greeting in ['hi', 'hello', 'hey', 'thanks', 'thank you']):
                # Try to search documents first
                try:
                    search_result = await tool_caller.search_documents(message, limit=2)
                    
                    # Create context-aware prompt with better formatting
                    base_prompt = f"""You are helping a user search through their personal notes. Based on their question: "{message}"

Here's what I found in their documents:
{search_result}

Please format your response with this structure:

## Answer
[Provide a clear, direct answer to their question using the information found]

## References
[List the source documents with brief descriptions of relevant content, using 📄 emoji for each source]

## Additional Context
[Any extra relevant information or context that might be helpful, but keep this brief]

Guidelines:
- Give the direct answer first before anything else
- If no relevant information was found, simply say so in the Answer section
- Keep references clean and organized
- Be conversational but well-structured
- Use markdown formatting for better readability

User question: {message}"""
                    
                    # Provide explicit thinking instructions
                    context_prompt = base_prompt + """\n\nIMPORTANT: Use <think>...</think> tags to show your reasoning process before providing your final structured response. For example:

<think>
Let me analyze the search results:
1. What information is relevant...
2. How to structure the answer...
3. What references to include...
</think>

Then provide your structured response."""
                    
                    # Stream response with context
                    payload = {
                        "model": CHAT_MODEL,
                        "prompt": context_prompt,
                        "stream": True
                    }
                    
                    full_response = ""  # Track the complete response
                    async with client.stream('POST', f"http://{OLLAMA_HOST}/api/generate", json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'response' in data and data['response']:
                                        chunk = data['response']
                                        full_response += chunk
                                        yield chunk
                                    elif data.get('done', False):
                                        # Log the complete response when done
                                        logger.info(f"COMPLETE QWEN RESPONSE:\n{full_response}")
                                        logger.info(f"Response contains <think>: {'<think>' in full_response}")
                                        logger.info(f"Response contains thinking: {'thinking' in full_response.lower()}")
                                except json.JSONDecodeError:
                                    continue
                                    
                except Exception as search_error:
                    logger.error(f"Search failed: {search_error}")
                    # Fallback to normal chat
                    async for chunk in stream_normal_response(client, message):
                        yield chunk
            else:
                # Normal chat without tools
                async for chunk in stream_normal_response(client, message):
                    yield chunk
                
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        yield f"Sorry, I encountered an error: {str(e)}"

async def stream_normal_response(client: httpx.AsyncClient, message: str):
    """Stream a normal chat response without tools"""
    base_context = """You are a helpful assistant for a personal note-taking system. 
    
You can help users search through their notes and documents. When they ask about specific topics, projects, or information, let them know you can search their documents.

For general conversation, respond naturally and helpfully."""
    
    # Provide explicit thinking instructions for all models
    system_context = base_context + """\n\nIMPORTANT: For any question that requires analysis or reasoning, you MUST use <think>...</think> tags to show your thought process before providing your final answer. For example:

<think>
Let me think about this step by step:
1. First consideration...
2. Second point...
3. Conclusion...
</think>

Then provide your actual response."""
    
    payload = {
        "model": CHAT_MODEL,
        "prompt": f"System: {system_context}\n\nUser: {message}\n\nAssistant:",
        "stream": True
    }
    
    full_response = ""  # Track the complete response
    async with client.stream('POST', f"http://{OLLAMA_HOST}/api/generate", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data and data['response']:
                        chunk = data['response']
                        full_response += chunk
                        yield chunk
                    elif data.get('done', False):
                        # Log the complete response when done
                        logger.info(f"COMPLETE QWEN NORMAL RESPONSE:\n{full_response}")
                        logger.info(f"Response contains <think>: {'<think>' in full_response}")
                        logger.info(f"Response contains thinking: {'thinking' in full_response.lower()}")
                except json.JSONDecodeError:
                    continue

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message.strip():
                continue
                
            logger.info(f"Received message: {user_message}")
            
            # Send response chunks
            async for chunk in chat_with_ollama(user_message):
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": chunk
                }))
            
            # Signal end of response
            await websocket.send_text(json.dumps({
                "type": "end"
            }))
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/")
async def get_chat_ui(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_detailed_status():
    """Get detailed system status"""
    try:
        async with httpx.AsyncClient() as client:
            # Get processor status
            processor_response = await client.get(f"http://{PROCESSOR_HOST}/status")
            processor_data = processor_response.json() if processor_response.status_code == 200 else {}
            
            # Test Ollama connection
            try:
                ollama_response = await client.get(f"http://{OLLAMA_HOST}/api/tags")
                ollama_status = "connected" if ollama_response.status_code == 200 else "disconnected"
                ollama_models = ollama_response.json().get("models", []) if ollama_response.status_code == 200 else []
            except:
                ollama_status = "disconnected"
                ollama_models = []
            
            return {
                "service": "chat-interface",
                "status": "healthy",
                "processor": {
                    "status": "connected" if processor_response.status_code == 200 else "disconnected",
                    "documents_indexed": processor_data.get("documents_indexed", 0),
                    "embedding_model": processor_data.get("embedding_model", "Unknown"),
                    "total_chunks": processor_data.get("total_chunks", 0),
                    "collection_name": processor_data.get("collection_name", "Unknown")
                },
                "ollama": {
                    "status": ollama_status,
                    "host": OLLAMA_HOST,
                    "chat_model": CHAT_MODEL,
                    "available_models": [model.get("name", "") for model in ollama_models]
                }
            }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "service": "chat-interface", 
            "status": "error",
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chat-interface"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")