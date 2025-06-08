# Docker Deployment Guide

This guide explains how to run the Image Editing Assistant using Docker.

## Prerequisites

- Docker and Docker Compose installed
- Gemini API key from Google AI Studio

## Quick Start

### 1. Set up environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### 3. Using Docker directly

```bash
# Build the image
docker build -t image-editing-assistant .

# Run the container
docker run -d \
  --name image-editing-assistant \
  -p 7860:7860 \
  -e GEMINI_API_KEY=your_api_key_here \
  -v $(pwd)/logs:/app/logs \
  image-editing-assistant
```

## Accessing the Application

Once running, access the application at:

- **Local**: <http://localhost:7860>
- **Network**: <http://your-server-ip:7860>

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `GRADIO_SERVER_NAME`: Server bind address (default: 0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (default: 7860)

### Volumes

- `./logs:/app/logs`: Persist application logs
- `./test_images:/app/test_images:ro`: Mount test images (read-only)

## Production Deployment

For production deployment:

1. Use a reverse proxy (nginx/traefik) for HTTPS
2. Set up proper logging and monitoring
3. Configure resource limits:

```yaml
services:
  image-editing-assistant:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G
```

## Troubleshooting

### Container won't start

- Check if port 7860 is available
- Verify your API key is set correctly
- Check container logs: `docker-compose logs`

### Out of memory errors

- Increase Docker memory limits
- Monitor resource usage: `docker stats`

### Permission issues

- Ensure the logs directory is writable
- Check file permissions on mounted volumes

## Development

For development with live code reloading:

```bash
# Mount source code as volume
docker run -d \
  --name image-editing-assistant-dev \
  -p 7860:7860 \
  -e GEMINI_API_KEY=your_api_key_here \
  -v $(pwd):/app \
  -v $(pwd)/logs:/app/logs \
  image-editing-assistant
```

## Health Checks

The container includes health checks that verify the application is responding:

- Endpoint: `http://localhost:7860/`
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
