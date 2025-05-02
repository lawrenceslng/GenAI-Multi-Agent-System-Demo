# FROM python:3.10-slim

# # Create workspace directory for temporary code storage
# WORKDIR /app

# # Create sandbox and workspace directories
# RUN mkdir -p sandbox workspace

# # Copy agent code and requirements
# COPY docker_code_agent.py /app/docker_code_agent.py
# COPY requirements.txt /app/requirements.txt

# # Install dependencies
# RUN pip install --no-cache-dir -r /app/requirements.txt

# # Create non-root user for security
# RUN useradd -m -u 1000 codeagent && \
#     chown -R codeagent:codeagent /app

# # Switch to non-root user
# USER codeagent

# # Set entrypoint
# ENTRYPOINT ["python", "/app/docker_code_agent.py"]

FROM python:3.10-slim

# Create workspace directory for temporary code storage
WORKDIR /workspace

# Copy agent code and requirements
COPY docker_code_agent.py /app/docker_code_agent.py
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 codeagent && \
    chown -R codeagent:codeagent /workspace

# Switch to non-root user
USER codeagent

# Set entrypoint
ENTRYPOINT ["python", "/app/docker_code_agent.py"]