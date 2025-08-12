class ChatClient {
    constructor() {
        this.socket = null;
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.messages = document.getElementById('messages');
        this.status = document.getElementById('status');
        this.modelStatus = document.getElementById('modelStatus');
        this.docStatus = document.getElementById('docStatus');
        this.currentResponse = null;
        this.responseBuffer = '';
        this.thinkingBuffer = '';
        this.isInThinking = false;
        this.hasThinking = false;
        
        // Configure marked.js for safe markdown rendering
        marked.setOptions({
            breaks: true,
            gfm: true,
            sanitize: false // We'll use DOMPurify instead
        });
        
        this.connect();
        this.setupEventListeners();
        this.loadSystemStatus();
        
        // Load status every 30 seconds
        setInterval(() => this.loadSystemStatus(), 30000);
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
        
        // Show typing indicator and reset streaming state
        this.currentResponse = this.addMessage('Thinking...', 'assistant');
        this.currentResponse.classList.add('typing-indicator');
        this.responseBuffer = '';
        this.thinkingBuffer = '';
        this.isInThinking = false;
        this.hasThinking = false;
        
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
                // Process the chunk for thinking tags
                this.processStreamingChunk(data.content);
            }
        } else if (data.type === 'end') {
            // Final render with thinking support when stream is complete
            if (this.currentResponse) {
                this.renderFinalResponse();
            }
            
            this.resetStreamingState();
        }
    }
    
    processStreamingChunk(chunk) {
        let remainingChunk = chunk;
        
        while (remainingChunk.length > 0) {
            if (!this.isInThinking) {
                // Look for start of thinking tag
                const thinkStart = remainingChunk.indexOf('<think>');
                if (thinkStart !== -1) {
                    // Add content before thinking tag to response
                    if (thinkStart > 0) {
                        this.responseBuffer += remainingChunk.substring(0, thinkStart);
                        this.renderMarkdownResponse(this.responseBuffer);
                    }
                    
                    // Enter thinking mode
                    this.isInThinking = true;
                    this.hasThinking = true;
                    this.showThinkingIndicator();
                    
                    // Continue with content after the tag
                    remainingChunk = remainingChunk.substring(thinkStart + 7); // 7 = '<think>'.length
                } else {
                    // No thinking tag, add to response buffer
                    this.responseBuffer += remainingChunk;
                    this.renderMarkdownResponse(this.responseBuffer);
                    break;
                }
            } else {
                // We're in thinking mode, look for end tag
                const thinkEnd = remainingChunk.indexOf('</think>');
                if (thinkEnd !== -1) {
                    // Add content before end tag to thinking buffer
                    this.thinkingBuffer += remainingChunk.substring(0, thinkEnd);
                    
                    // Exit thinking mode
                    this.isInThinking = false;
                    this.hideThinkingIndicator();
                    
                    // Continue with content after the tag
                    remainingChunk = remainingChunk.substring(thinkEnd + 8); // 8 = '</think>'.length
                } else {
                    // Still in thinking, add to thinking buffer
                    this.thinkingBuffer += remainingChunk;
                    break;
                }
            }
        }
        
        this.scrollToBottom();
    }
    
    showThinkingIndicator() {
        if (this.currentResponse && !this.currentResponse.querySelector('.thinking-indicator-live')) {
            const indicator = document.createElement('div');
            indicator.className = 'thinking-indicator-live';
            indicator.innerHTML = '🤔 <em>Thinking...</em>';
            indicator.style.cssText = 'color: #666; font-style: italic; margin-bottom: 10px;';
            this.currentResponse.appendChild(indicator);
        }
    }
    
    hideThinkingIndicator() {
        if (this.currentResponse) {
            const indicator = this.currentResponse.querySelector('.thinking-indicator-live');
            if (indicator) {
                indicator.remove();
            }
        }
    }
    
    renderFinalResponse() {
        if (this.hasThinking && this.thinkingBuffer) {
            // Create thinking + response structure
            this.currentResponse.classList.add('has-thinking');
            
            const thinkingSection = this.createThinkingSection(this.thinkingBuffer);
            const responseSection = this.createResponseSection(this.responseBuffer);
            
            this.currentResponse.innerHTML = '';
            this.currentResponse.appendChild(thinkingSection);
            this.currentResponse.appendChild(responseSection);
        } else {
            // No thinking content, just render the response
            this.renderMarkdownResponse(this.responseBuffer);
        }
    }
    
    resetStreamingState() {
        this.currentResponse = null;
        this.responseBuffer = '';
        this.thinkingBuffer = '';
        this.isInThinking = false;
        this.hasThinking = false;
        this.sendButton.disabled = false;
        this.messageInput.focus();
    }
    
    renderMarkdownResponse(content) {
        if (!this.currentResponse) return;
        
        try {
            // Convert markdown to HTML
            const rawHtml = marked.parse(content);
            
            // Sanitize the HTML to prevent XSS
            const cleanHtml = DOMPurify.sanitize(rawHtml);
            
            // Set the sanitized HTML
            this.currentResponse.innerHTML = cleanHtml;
        } catch (error) {
            console.error('Error rendering markdown:', error);
            // Fallback to plain text
            this.currentResponse.textContent = content;
        }
    }
    
    renderResponseWithThinking(content) {
        if (!this.currentResponse) return;
        
        // Parse thinking content from response
        const { thinking, response } = this.parseThinkingContent(content);
        
        console.log('Thinking parsed:', thinking ? 'YES' : 'NO');
        
        if (thinking) {
            // Create structure for thinking + response
            this.currentResponse.classList.add('has-thinking');
            
            const thinkingSection = this.createThinkingSection(thinking);
            const responseSection = this.createResponseSection(response);
            
            this.currentResponse.innerHTML = '';
            this.currentResponse.appendChild(thinkingSection);
            this.currentResponse.appendChild(responseSection);
        } else {
            // No thinking content, render normally
            this.renderMarkdownResponse(content);
        }
    }
    
    parseThinkingContent(content) {
        // Handle null/undefined content
        if (!content) {
            console.log('parseThinkingContent: content is null/undefined');
            return { thinking: null, response: '' };
        }
        
        
        // Look for thinking content in XML-style tags
        const patterns = [
            /<thinking>(.*?)<\/thinking>/gs,
            /<think>(.*?)<\/think>/gs,
            /\[THINKING\](.*?)\[\/THINKING\]/gs,
            /\[thinking\](.*?)\[\/thinking\]/gs
        ];
        
        for (let i = 0; i < patterns.length; i++) {
            const pattern = patterns[i];
            const match = content.match(pattern);
            if (match && match[1] !== undefined) {
                const thinking = match[1].trim();
                const response = content.replace(match[0], '').trim();
                console.log('Successfully parsed thinking content with pattern', i + 1);
                console.log('Thinking length:', thinking.length);
                console.log('Response length:', response.length);
                return { thinking, response };
            }
        }
        
        return { thinking: null, response: content };
    }
    
    createThinkingSection(thinking) {
        const section = document.createElement('div');
        section.className = 'thinking-section';
        
        const header = document.createElement('div');
        header.className = 'thinking-header';
        header.innerHTML = `
            <span class="thinking-toggle">▶</span>
            <span>🤔 Model's thinking process</span>
        `;
        
        const content = document.createElement('div');
        content.className = 'thinking-content';
        content.textContent = thinking;
        
        // Add click handler for toggle
        header.addEventListener('click', () => {
            const toggle = header.querySelector('.thinking-toggle');
            const isExpanded = content.classList.contains('expanded');
            
            if (isExpanded) {
                content.classList.remove('expanded');
                toggle.classList.remove('expanded');
                toggle.textContent = '▶';
            } else {
                content.classList.add('expanded');
                toggle.classList.add('expanded');
                toggle.textContent = '▼';
            }
        });
        
        section.appendChild(header);
        section.appendChild(content);
        
        return section;
    }
    
    createResponseSection(response) {
        const section = document.createElement('div');
        section.className = 'response-content';
        
        try {
            // Convert markdown to HTML
            const rawHtml = marked.parse(response);
            
            // Sanitize the HTML to prevent XSS
            const cleanHtml = DOMPurify.sanitize(rawHtml);
            
            // Set the sanitized HTML
            section.innerHTML = cleanHtml;
        } catch (error) {
            console.error('Error rendering markdown:', error);
            // Fallback to plain text
            section.textContent = response;
        }
        
        return section;
    }
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        if (sender === 'user') {
            // User messages are plain text
            messageDiv.textContent = content;
        } else {
            // Assistant messages support markdown and thinking
            if (content === 'Thinking...') {
                messageDiv.textContent = content;
            } else {
                const { thinking, response } = this.parseThinkingContent(content);
                
                if (thinking) {
                    messageDiv.classList.add('has-thinking');
                    
                    const thinkingSection = this.createThinkingSection(thinking);
                    const responseSection = this.createResponseSection(response);
                    
                    messageDiv.appendChild(thinkingSection);
                    messageDiv.appendChild(responseSection);
                } else {
                    this.renderMarkdownInElement(messageDiv, content);
                }
            }
        }
        
        this.messages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    renderMarkdownInElement(element, content) {
        try {
            // Convert markdown to HTML
            const rawHtml = marked.parse(content);
            
            // Sanitize the HTML to prevent XSS
            const cleanHtml = DOMPurify.sanitize(rawHtml);
            
            // Set the sanitized HTML
            element.innerHTML = cleanHtml;
        } catch (error) {
            console.error('Error rendering markdown:', error);
            // Fallback to plain text
            element.textContent = content;
        }
    }
    
    scrollToBottom() {
        this.messages.scrollTop = this.messages.scrollHeight;
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            this.updateStatusBadges(data);
        } catch (error) {
            console.error('Error loading system status:', error);
            this.modelStatus.textContent = 'Model: Error';
            this.docStatus.textContent = 'Docs: Error';
        }
    }
    
    updateStatusBadges(data) {
        // Update model status
        const chatModel = data.ollama?.chat_model || 'Unknown';
        const modelName = chatModel.split(':')[0]; // Extract just the model name (e.g., "llama3.1" from "llama3.1:8b")
        this.modelStatus.textContent = `Model: ${modelName}`;
        
        // Update document count
        const docCount = data.processor?.documents_indexed || 0;
        this.docStatus.textContent = `Docs: ${docCount}`;
        
        // Add healthy class if everything looks good
        const isHealthy = data.ollama?.status === 'connected' && data.processor?.status === 'connected';
        this.modelStatus.className = `status-badge ${isHealthy ? 'healthy' : ''}`;
        this.docStatus.className = `status-badge ${isHealthy ? 'healthy' : ''}`;
    }
}

// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatClient();
});