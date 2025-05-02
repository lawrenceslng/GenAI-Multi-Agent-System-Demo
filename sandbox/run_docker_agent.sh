# #!/bin/bash

# # Exit on error
# set -e

# echo "🐳 Starting Docker Code Agent..."
# echo "================================"

# # Build the Docker image
# echo "🏗️  Building Docker image..."
# docker build -t docker_code_agent ./sandbox

# # Run the container
# echo "🚀 Launching Docker container..."
# echo "--------------------------------"
# echo "📁 Mounting sandbox volume as read-only"
# echo "🔒 Running as non-root user (UID: 1000)"
# echo "🌐 Passing environment variables"
# echo "--------------------------------"

# # Create a temporary directory for the container's workspace
# TEMP_DIR=$(mktemp -d)
# echo "📂 Created temporary workspace: ${TEMP_DIR}"

# # Run container with:
# # 1. Sandbox mounted as read-only in /app/sandbox
# # 2. Temporary workspace directory mounted in /app/workspace
# # 3. Environment variables passed through
# # 4. Non-root user
# docker run --rm \
#     --mount type=bind,source="$(pwd)/sandbox",target=/app/sandbox,readonly \
#     --mount type=bind,source="${TEMP_DIR}",target=/app/workspace \
#     -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
#     -e GITHUB_PAT="${GITHUB_PAT}" \
#     -e GITHUB_MCP_URL="${GITHUB_MCP_URL}" \
#     --user 1000:1000 \
#     docker_code_agent

# # Clean up temporary directory
# echo "🧹 Cleaning up temporary workspace..."
# rm -rf "${TEMP_DIR}"

# echo "✨ Docker container completed execution"

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