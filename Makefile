# Makefile for Local RAG Chat System
# Automatically detects OS and uses appropriate docker-compose configuration

# Detect OS
UNAME_S := $(shell uname -s)

# Set docker-compose command based on OS
ifeq ($(UNAME_S),Linux)
    COMPOSE_CMD = docker-compose -f docker-compose.yml -f docker-compose.linux.yml
else
    COMPOSE_CMD = docker-compose
endif

.PHONY: help up down logs logs-processor logs-chat logs-mcp build rebuild status reset process-all search clean

# Default target
help:
	@echo "Local RAG Chat System - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  up              - Start all services (OS-aware)"
	@echo "  down            - Stop all services"
	@echo "  logs            - Show logs for all services"
	@echo "  logs-processor  - Show logs for processor service"
	@echo "  logs-chat       - Show logs for chat-interface service"
	@echo "  logs-mcp        - Show logs for mcp-server service"
	@echo "  build           - Build all services"
	@echo "  rebuild         - Rebuild and restart all services"
	@echo "  status          - Check system status"
	@echo "  reset           - Reset document collection"
	@echo "  process-all     - Process all documents"
	@echo "  search          - Interactive search (requires QUERY variable)"
	@echo "  clean           - Stop and remove all containers and volumes"
	@echo ""
	@echo "OS detected: $(UNAME_S)"
	@echo "Using: $(COMPOSE_CMD)"

# Start services
up:
	@echo "Starting services on $(UNAME_S)..."
	$(COMPOSE_CMD) up -d

# Stop services
down:
	@echo "Stopping services..."
	$(COMPOSE_CMD) down

# Show logs
logs:
	$(COMPOSE_CMD) logs -f

logs-processor:
	$(COMPOSE_CMD) logs -f processor

logs-chat:
	$(COMPOSE_CMD) logs -f chat-interface

logs-mcp:
	$(COMPOSE_CMD) logs -f mcp-server

# Build services
build:
	$(COMPOSE_CMD) build

# Rebuild and restart
rebuild:
	@echo "Rebuilding and restarting services..."
	$(COMPOSE_CMD) build
	$(COMPOSE_CMD) up -d

# Check status
status:
	@echo "Checking system status..."
	@curl -s http://localhost:8001/status || echo "Processor service not responding"

# Reset document collection
reset:
	@echo "Resetting document collection..."
	@curl -X POST http://localhost:8001/reset

# Process all documents
process-all:
	@echo "Processing all documents..."
	@curl -X POST http://localhost:8001/process-all

# Search documents (usage: make search QUERY="your search terms")
search:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make search QUERY=\"your search terms\""; \
	else \
		curl -X POST http://localhost:8001/search \
			-H "Content-Type: application/json" \
			-d '{"query": "$(QUERY)", "limit": 5}'; \
	fi

# Clean up everything
clean:
	@echo "Stopping and removing all containers and volumes..."
	$(COMPOSE_CMD) down -v --remove-orphans