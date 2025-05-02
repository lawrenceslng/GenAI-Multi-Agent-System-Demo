#!/bin/bash

# Exit on error
set -e

echo "🐳 Starting Docker Code Agent..."
echo "================================"

# Build the Docker image
echo "🏗️  Building Docker image..."
docker build -t docker_code_agent ./sandbox

# Run the container
echo "🚀 Launching Docker container..."
echo "--------------------------------"
echo "📁 Mounting sandbox volume as read-only"
echo "🔒 Running as non-root user (UID: 1000)"
echo "🌐 Passing environment variables"
echo "--------------------------------"

docker run --rm \
    --mount type=bind,source="$(pwd)/sandbox",target=/sandbox,readonly \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    -e GITHUB_PAT="${GITHUB_PAT}" \
    -e GITHUB_MCP_URL="${GITHUB_MCP_URL}" \
    --user 1000:1000 \
    docker_code_agent

echo "✨ Docker container completed execution"