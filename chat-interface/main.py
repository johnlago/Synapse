#!/usr/bin/env python3
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import ollama
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-interface")

app = FastAPI(title="Local RAG Chat Interface")

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "host.docker.internal:11434")
PROCESSOR_HOST = os.getenv("PROCESSOR_HOST", "processor:8001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")

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
    
    async def search_documents(self, query: str, limit: int = 5) -> str:
        """Search documents via processor API"""
        try:
            url = f"http://{PROCESSOR_HOST}/search"
            payload = {"query": query, "limit": limit}
            
            response = await self.http_client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if "error" in data:
                return f"Search error: {data['error']}"
            
            results = data.get("results", [])
            if not results:
                return "No relevant documents found for your query."
            
            # Format results for chat
            formatted = f"Found {len(results)} relevant documents:\n\n"
            for i, result in enumerate(results, 1):
                formatted += f"**Result {i}** (similarity: {result.get('similarity', 0):.2f})\n"
                formatted += f"Source: {result.get('source', 'Unknown')}\n"
                formatted += f"Content: {result.get('content', '')[:400]}...\n\n"
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return f"Error searching documents: {str(e)}"
    
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
    """Stream chat response from Ollama with tool calling"""
    try:
        client = ollama.AsyncClient(host=f"http://{OLLAMA_HOST}")
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that can search through the user's personal notes and documents. 

When the user asks questions about their notes, projects, or any topic that might be in their documents, use the search_documents tool to find relevant information.

Always provide helpful, conversational responses based on the search results. If you find relevant information, summarize it naturally and mention the sources.

You can also check the status of the document collection if asked about what's available or indexed."""
            },
            {
                "role": "user", 
                "content": message
            }
        ]
        
        # First call to get potential tool calls
        response = await client.chat(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            stream=False
        )
        
        # Check if there are tool calls
        if response['message'].get('tool_calls'):
            # Execute tool calls
            for tool_call in response['message']['tool_calls']:
                function = tool_call['function']
                tool_result = await call_tool(function['name'], function['arguments'])
                
                # Add tool result to conversation
                messages.append(response['message'])
                messages.append({
                    "role": "tool",
                    "content": tool_result
                })
            
            # Get final response with tool results
            final_response = client.chat(
                model=CHAT_MODEL,
                messages=messages,
                stream=True
            )
            
            async for chunk in final_response:
                if chunk['message']['content']:
                    yield chunk['message']['content']
        else:
            # No tools needed, stream the response
            response_stream = client.chat(
                model=CHAT_MODEL,
                messages=messages,
                stream=True
            )
            
            async for chunk in response_stream:
                if chunk['message']['content']:
                    yield chunk['message']['content']
                    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        yield f"Error: {str(e)}"

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
async def get_chat_ui():
    """Serve the chat interface"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local RAG Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: #2c3e50;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message {
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            word-wrap: break-word;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            align-self: flex-end;
        }
        
        .assistant-message {
            background: #e9ecef;
            color: #333;
            align-self: flex-start;
            white-space: pre-wrap;
        }
        
        .input-container {
            display: flex;
            padding: 1rem;
            gap: 0.5rem;
            border-top: 1px solid #dee2e6;
        }
        
        .input-field {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid #dee2e6;
            border-radius: 0.5rem;
            font-size: 1rem;
        }
        
        .send-button {
            padding: 0.75rem 1.5rem;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 1rem;
        }
        
        .send-button:hover {
            background: #0056b3;
        }
        
        .send-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .typing-indicator {
            background: #e9ecef;
            color: #666;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            align-self: flex-start;
            font-style: italic;
        }
        
        .status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }
        
        .status.connected {
            background: #d4edda;
            color: #155724;
        }
        
        .status.disconnected {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Local RAG Chat</h1>
        <p>Ask questions about your notes and documents</p>
    </div>
    
    <div class="status disconnected" id="status">Connecting...</div>
    
    <div class="chat-container">
        <div class="messages" id="messages">
            <div class="message assistant-message">
                Hello! I can help you search through your personal notes and documents. Ask me anything about your projects, ideas, or any topic you've written about.
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" 
                   class="input-field" 
                   id="messageInput" 
                   placeholder="Ask about your notes..."
                   autocomplete="off">
            <button class="send-button" id="sendButton">Send</button>
        </div>
    </div>

    <script>
        class ChatClient {
            constructor() {
                this.socket = null;
                this.messageInput = document.getElementById('messageInput');
                this.sendButton = document.getElementById('sendButton');
                this.messages = document.getElementById('messages');
                this.status = document.getElementById('status');
                this.currentResponse = null;
                
                this.connect();
                this.setupEventListeners();
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/chat`;
                
                this.socket = new WebSocket(wsUrl);
                
                this.socket.onopen = () => {
                    this.status.textContent = 'Connected';
                    this.status.className = 'status connected';
                };
                
                this.socket.onclose = () => {
                    this.status.textContent = 'Disconnected';
                    this.status.className = 'status disconnected';
                    setTimeout(() => this.connect(), 3000);
                };
                
                this.socket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.socket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            
            setupEventListeners() {
                this.sendButton.addEventListener('click', () => this.sendMessage());
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
            }
            
            sendMessage() {
                const message = this.messageInput.value.trim();
                if (!message || this.socket.readyState !== WebSocket.OPEN) return;
                
                // Add user message to chat
                this.addMessage(message, 'user');
                
                // Clear input and disable
                this.messageInput.value = '';
                this.sendButton.disabled = true;
                
                // Show typing indicator
                this.currentResponse = this.addMessage('Thinking...', 'assistant');
                this.currentResponse.classList.add('typing-indicator');
                
                // Send to server
                this.socket.send(JSON.stringify({ message }));
            }
            
            handleMessage(data) {
                if (data.type === 'chunk') {
                    if (this.currentResponse && this.currentResponse.classList.contains('typing-indicator')) {
                        this.currentResponse.textContent = '';
                        this.currentResponse.classList.remove('typing-indicator');
                    }
                    
                    if (this.currentResponse) {
                        this.currentResponse.textContent += data.content;
                        this.scrollToBottom();
                    }
                } else if (data.type === 'end') {
                    this.currentResponse = null;
                    this.sendButton.disabled = false;
                    this.messageInput.focus();
                }
            }
            
            addMessage(content, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.textContent = content;
                
                this.messages.appendChild(messageDiv);
                this.scrollToBottom();
                
                return messageDiv;
            }
            
            scrollToBottom() {
                this.messages.scrollTop = this.messages.scrollHeight;
            }
        }
        
        // Initialize chat when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new ChatClient();
        });
    </script>
</body>
</html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chat-interface"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")