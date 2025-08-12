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
        
        // Show typing indicator
        this.currentResponse = this.addMessage('Thinking...', 'assistant');
        this.currentResponse.classList.add('typing-indicator');
        this.responseBuffer = '';
        
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
                // Accumulate the response buffer
                this.responseBuffer += data.content;
                
                // Render with thinking support in real-time
                this.renderResponseWithThinking(this.responseBuffer);
                this.scrollToBottom();
            }
        } else if (data.type === 'end') {
            // Final render to ensure everything is properly formatted
            if (this.currentResponse && this.responseBuffer) {
                this.renderResponseWithThinking(this.responseBuffer);
            }
            
            this.currentResponse = null;
            this.responseBuffer = '';
            this.sendButton.disabled = false;
            this.messageInput.focus();
        }
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
        
        // Debug logging
        console.log('Content length:', content?.length || 0);
        console.log('Has thinking:', !!thinking);
        if (thinking) {
            console.log('Thinking preview:', thinking.substring(0, 100));
        }
        
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
            return { thinking: null, response: '' };
        }
        
        // Look for thinking content in various formats
        const patterns = [
            // XML-style tags
            /<thinking>(.*?)<\/thinking>/gs,
            /<think>(.*?)<\/think>/gs,
            /\[THINKING\](.*?)\[\/THINKING\]/gs,
            /\[thinking\](.*?)\[\/thinking\]/gs,
            // Qwen's format: "Thinking..." followed by "...done thinking."
            /Thinking\.\.\.(.*?)\.\.\.done thinking\./gs,
            // Alternative formats
            /^Thinking:\s*(.*?)(?=\n\n|\n[A-Z]|$)/gs,
            /\*thinking\*(.*?)\*\/thinking\*/gs
        ];
        
        for (const pattern of patterns) {
            const match = content.match(pattern);
            if (match && match[1]) {
                const thinking = match[1].trim();
                const response = content.replace(pattern, '').trim();
                return { thinking, response };
            }
        }
        
        // Try to detect Qwen's natural thinking pattern
        // Look for thinking content followed by structured sections like "Answer", "## Answer", etc.
        const structuredSectionPattern = /^(.*?)(?=\n(?:Answer|## Answer|# Answer|\*\*Answer\*\*|References|## References))/s;
        const match = content.match(structuredSectionPattern);
        
        if (match && match[1]) {
            const potentialThinking = match[1].trim();
            const remainingContent = content.substring(match[0].length).trim();
            
            // Check if the potential thinking section looks like actual thinking
            // (contains reasoning words, is longer than a simple sentence, etc.)
            const thinkingIndicators = [
                'let me', 'first', 'second', 'i need', 'i should', 'the user', 'this means',
                'result 1', 'result 2', 'result 3', 'similarity', 'relevant', 'seems',
                'so the', 'therefore', 'however', 'but', 'because', 'since', 'given'
            ];
            
            const hasThinkingIndicators = thinkingIndicators.some(indicator => 
                potentialThinking.toLowerCase().includes(indicator)
            );
            
            // Only treat as thinking if it's substantial and has reasoning indicators
            if (potentialThinking.length > 50 && hasThinkingIndicators && remainingContent.length > 0) {
                return { thinking: potentialThinking, response: remainingContent };
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