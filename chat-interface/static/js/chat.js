class ChatClient {
    constructor() {
        this.socket = null;
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.messages = document.getElementById('messages');
        this.status = document.getElementById('status');
        this.statusDashboard = document.getElementById('statusDashboard');
        this.statusContent = document.getElementById('statusContent');
        this.toggleStatusBtn = document.getElementById('toggleStatus');
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
        
        // Status dashboard toggle
        this.toggleStatusBtn.addEventListener('click', () => this.toggleStatusDashboard());
        document.querySelector('.status-header').addEventListener('click', () => this.toggleStatusDashboard());
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
                
                // Render markdown in real-time
                this.renderMarkdownResponse(this.responseBuffer);
                this.scrollToBottom();
            }
        } else if (data.type === 'end') {
            // Final render to ensure everything is properly formatted
            if (this.currentResponse && this.responseBuffer) {
                this.renderMarkdownResponse(this.responseBuffer);
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
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        if (sender === 'user') {
            // User messages are plain text
            messageDiv.textContent = content;
        } else {
            // Assistant messages support markdown
            if (content === 'Thinking...') {
                messageDiv.textContent = content;
            } else {
                this.renderMarkdownInElement(messageDiv, content);
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
            this.renderStatusDashboard(data);
        } catch (error) {
            console.error('Error loading system status:', error);
            this.statusContent.innerHTML = '<div class="status-loading">Error loading system status</div>';
        }
    }
    
    renderStatusDashboard(data) {
        const statusGrid = document.createElement('div');
        statusGrid.className = 'status-grid';
        
        // Processor status
        const processorCard = this.createStatusCard(
            'Document Processor',
            data.processor?.status || 'unknown',
            [
                `Documents: ${data.processor?.documents_indexed || 0}`,
                `Chunks: ${data.processor?.total_chunks || 0}`,
                `Model: ${data.processor?.embedding_model || 'Unknown'}`
            ]
        );
        
        // Ollama status
        const ollamaCard = this.createStatusCard(
            'Ollama AI',
            data.ollama?.status || 'unknown',
            [
                `Chat Model: ${data.ollama?.chat_model || 'Unknown'}`,
                `Available Models: ${data.ollama?.available_models?.length || 0}`,
                `Host: ${data.ollama?.host || 'Unknown'}`
            ]
        );
        
        // Chat interface status
        const chatCard = this.createStatusCard(
            'Chat Interface',
            data.status || 'unknown',
            [
                `Service: ${data.service || 'Unknown'}`,
                `WebSocket: ${this.socket?.readyState === WebSocket.OPEN ? 'Connected' : 'Disconnected'}`,
                `Last Updated: ${new Date().toLocaleTimeString()}`
            ]
        );
        
        statusGrid.appendChild(processorCard);
        statusGrid.appendChild(ollamaCard);
        statusGrid.appendChild(chatCard);
        
        this.statusContent.innerHTML = '';
        this.statusContent.appendChild(statusGrid);
    }
    
    createStatusCard(title, status, details) {
        const card = document.createElement('div');
        card.className = `status-card ${status === 'connected' || status === 'healthy' ? 'connected' : 
                                       status === 'disconnected' || status === 'error' ? 'disconnected' : 'warning'}`;
        
        card.innerHTML = `
            <h4>${title}</h4>
            <div class="status-value">${status.charAt(0).toUpperCase() + status.slice(1)}</div>
            ${details.map(detail => `<div class="status-detail">${detail}</div>`).join('')}
        `;
        
        return card;
    }
    
    toggleStatusDashboard() {
        const isCollapsed = this.statusContent.classList.contains('collapsed');
        
        if (isCollapsed) {
            this.statusContent.classList.remove('collapsed');
            this.toggleStatusBtn.classList.remove('collapsed');
            this.toggleStatusBtn.textContent = '▼';
        } else {
            this.statusContent.classList.add('collapsed');
            this.toggleStatusBtn.classList.add('collapsed');
            this.toggleStatusBtn.textContent = '▶';
        }
    }
}

// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatClient();
});