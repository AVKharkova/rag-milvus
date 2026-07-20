import os

compose_path = 'C:/Asias/Polyus/rag-milvus/docker-compose.yml'
with open(compose_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'langfuse-server' not in content:
    langfuse_config = """

  # --- Langfuse (Local Telemetry) ---
  langfuse-server:
    image: langfuse/langfuse:2
    container_name: rag-milvus-langfuse
    depends_on:
      langfuse-db:
        condition: service_healthy
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@langfuse-db:5432/postgres
      - NEXTAUTH_SECRET=mysecret
      - SALT=mysalt
      - ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000
      - TELEMETRY_ENABLED=false
      - LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false
    restart: unless-stopped

  langfuse-db:
    image: postgres:15
    container_name: rag-milvus-langfuse-db
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 2s
      timeout: 2s
      retries: 10
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=postgres
    volumes:
      - ./volumes/langfuse-db:/var/lib/postgresql/data
"""
    with open(compose_path, 'a', encoding='utf-8') as f:
        f.write(langfuse_config)
    print('Added Langfuse to docker-compose.yml')
else:
    print('Langfuse already in docker-compose.yml')
