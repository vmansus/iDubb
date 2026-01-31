#!/bin/bash

# NextGenMedia Docker Startup Script

set -e

echo "=== NextGenMedia Docker Startup ==="

cd "$(dirname "$0")/.."

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "Please edit .env with your platform credentials"
fi

# Build and start containers
echo ""
echo "Building and starting containers..."
docker-compose up --build -d

echo ""
echo "=== NextGenMedia is running! ==="
echo ""
echo "  Backend API:  http://localhost:8888"
echo "  Frontend:     http://localhost:3005"
echo "  API Docs:     http://localhost:8888/docs"
echo ""
echo "Commands:"
echo "  View logs:    docker-compose logs -f"
echo "  Stop:         docker-compose down"
echo "  Restart:      docker-compose restart"
echo ""
