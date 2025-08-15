# Linux-Specific Configuration

This document covers Linux-specific configuration and troubleshooting for the Local RAG system.

## Linux Networking Issues and Workarounds

### Ollama IPv6/IPv4 Binding Issue

On Linux systems, Ollama has a known networking issue where it doesn't properly bind to both IPv4 and IPv6:

- **Setting `OLLAMA_HOST=0.0.0.0`**: Makes Ollama listen only on IPv6, breaking IPv4 connectivity
- **Default configuration**: Makes Ollama listen only on loopback (127.0.0.1), preventing Docker container access

### Recommended Solution: Host Networking (Used by this project)

This project uses **host networking** for services that need Ollama access as the most reliable workaround:

**Advantages:**
- ✅ Bypasses Docker's network isolation issues with Ollama
- ✅ Works consistently across different Linux distributions
- ✅ No need to modify Ollama's systemd configuration
- ✅ Automatic OS detection via Makefile

**Security Considerations:**
- ⚠️ Services using host networking expose ports directly on your host
- ⚠️ Potential port conflicts with other services
- ⚠️ Reduced network isolation compared to bridge networking

**Ports exposed on host when using Linux configuration:**
- `8001` - Processor service
- `8002` - MCP server  
- `8003` - Chat interface
- ChromaDB remains isolated in bridge network

### Alternative: Manual Ollama Configuration (Not recommended)

If you prefer to modify Ollama instead of using host networking:

1. Edit Ollama service: `sudo vim /usr/lib/systemd/system/ollama.service`
2. Add: `Environment="OLLAMA_HOST=0.0.0.0"`
3. Reload and restart:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

**⚠️ Security Warning**: This exposes Ollama to your entire local network, allowing anyone on your network to access your Ollama instance if your firewall permits.

## Linux-Specific Commands

### Starting Services
```bash
# Use Linux-specific Docker Compose configuration
docker-compose -f docker-compose.yml -f docker-compose.linux.yml up -d

# Or use the Makefile (automatically detects Linux)
make up
```

### Troubleshooting

#### Check Ollama Service Status
```bash
sudo systemctl status ollama
```

#### Verify Ollama Network Binding
```bash
# Check what ports Ollama is listening on
sudo netstat -tlnp | grep ollama
# or
sudo ss -tlnp | grep ollama
```

#### Check Docker Host Connectivity
```bash
# Test connectivity from container to host
docker exec -it synapse-processor-1 curl http://host.docker.internal:11434/api/tags
```

#### Firewall Considerations
If you're using host networking, ensure your firewall allows the exposed ports:
```bash
# Example for ufw
sudo ufw allow 8001
sudo ufw allow 8002
sudo ufw allow 8003
```