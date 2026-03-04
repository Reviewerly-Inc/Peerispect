#!/bin/bash

# Docker setup script for Peerispect
# This script helps build and run the Docker containers

set -e

echo "🐳 Peerispect Docker Setup"
echo "=========================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     - Build the Docker image"
    echo "  up        - Start the services"
    echo "  down      - Stop the services"
    echo "  logs      - Show logs"
    echo "  restart   - Restart the services"
    echo "  clean     - Clean up containers and images"
    echo "  status    - Show container status"
    echo ""
}

# Function to build the image
build_image() {
    echo "🔨 Building Peerispect Docker image..."
    docker-compose build --no-cache
    echo "✅ Build completed!"
}

# Function to start services
start_services() {
    echo "🚀 Starting Peerispect services..."
    export USER_ID=$(id -u)
    export GROUP_ID=$(id -g)
    docker-compose up -d
    echo "✅ Services started!"
    echo ""
    echo "📋 Service URLs:"
    echo "  - Peerispect API: http://localhost:5015"
    echo "  - vLLM API: http://localhost:11435/v1"
    echo "  - API Documentation: http://localhost:5015/docs"
    echo ""
    echo "📊 Check status with: $0 status"
    echo "📝 View logs with: $0 logs"
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping Peerispect services..."
    docker-compose down
    echo "✅ Services stopped!"
}

# Function to show logs
show_logs() {
    echo "📝 Showing logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Function to restart services
restart_services() {
    echo "🔄 Restarting Peerispect services..."
    docker-compose restart
    echo "✅ Services restarted!"
}

# Function to clean up
clean_up() {
    echo "🧹 Cleaning up Docker resources..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    echo "✅ Cleanup completed!"
}

# Function to show status
show_status() {
    echo "📊 Container Status:"
    echo "==================="
    docker-compose ps
    echo ""
    echo "🔍 Health Check:"
    echo "==============="
    if curl -s http://localhost:5015/health > /dev/null 2>&1; then
        echo "✅ Peerispect API is healthy"
    else
        echo "❌ Peerispect API is not responding"
    fi
    
    if curl -s http://localhost:11435/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM API is healthy"
    else
        echo "❌ vLLM API is not responding"
    fi
}

# Main script logic
case "${1:-}" in
    build)
        build_image
        ;;
    up)
        start_services
        ;;
    down)
        stop_services
        ;;
    logs)
        show_logs
        ;;
    restart)
        restart_services
        ;;
    clean)
        clean_up
        ;;
    status)
        show_status
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
