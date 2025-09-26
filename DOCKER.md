# Docker Setup for Peerispect

This document explains how to run Peerispect using Docker.

## Quick Start

1. **Build and start the services:**
   ```bash
   ./docker-run.sh build
   ./docker-run.sh up
   ```

2. **Check if everything is running:**
   ```bash
   ./docker-run.sh status
   ```

3. **View logs:**
   ```bash
   ./docker-run.sh logs
   ```

## Available Commands

- `./docker-run.sh build` - Build the Docker image
- `./docker-run.sh up` - Start all services
- `./docker-run.sh down` - Stop all services
- `./docker-run.sh restart` - Restart services
- `./docker-run.sh logs` - View logs
- `./docker-run.sh status` - Check service status
- `./docker-run.sh clean` - Clean up containers and images

## Services

### Peerispect API
- **Port:** 5015
- **URL:** http://localhost:5015
- **Documentation:** http://localhost:5015/docs
- **Health Check:** http://localhost:5015/health

### Ollama
- **Port:** 11434
- **URL:** http://localhost:11434
- **Models:** Will be downloaded automatically when first used

## Configuration

The Docker setup uses the following configuration:

- **GPU Support:** Both services have access to NVIDIA GPUs
- **Volumes:** 
  - `./outputs` - For debugging (mounted to container)
  - `./api_cache` - For API caching (mounted to container)
  - `ollama_data` - For Ollama model storage (Docker volume)

## Environment Variables

- `OLLAMA_HOST=http://ollama:11434` - Ollama service URL
- `PYTHONPATH=/app` - Python path
- `PYTHONUNBUFFERED=1` - Python output buffering

## Integration with Existing Setup

This Docker setup is designed to work alongside your existing server setup. The Ollama service will be available at `http://localhost:11434` and can be used by other applications on the same network.

## Troubleshooting

1. **GPU not detected:**
   - Ensure NVIDIA Docker runtime is installed
   - Check that GPU devices are available: `nvidia-smi`

2. **Port conflicts:**
   - Change ports in `docker-compose.yml` if needed
   - Check what's using the ports: `netstat -tulpn | grep :5015`

3. **Permission issues:**
   - Ensure Docker has proper permissions
   - Run with `sudo` if needed

4. **Ollama models not loading:**
   - Check Ollama logs: `docker-compose logs ollama`
   - Pull models manually: `docker-compose exec ollama ollama pull qwen3:8b`

## Development

For development, you can mount the source code as a volume:

```yaml
volumes:
  - .:/app
```

This allows live code changes without rebuilding the image.

## Production Notes

- The current setup is minimal and suitable for development/testing
- For production, consider:
  - Using specific image tags instead of `latest`
  - Setting up proper secrets management
  - Configuring resource limits
  - Setting up monitoring and logging
  - Using external volumes for persistent data
