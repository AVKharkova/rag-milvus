# RAG Milvus

Облегчённый RAG-сервис **только на Milvus** (без OpenSearch / Postgres / dual-engine A/B).
Форк логики из `rag-platform`: hybrid dense+BM25 → RRF → опциональный rerank → `/v1/ask`.

## Что в стеке (облегчённые образы)

| Сервис | Образ | Зачем | ≈ RAM lim |
|--------|--------|--------|-----------|
| **etcd** | `quay.io/coreos/etcd:v3.5.18` | метаданные Milvus | 256 MB |
| **minio** | `minio/minio:RELEASE.2024-12-18…` | object store Milvus | 512 MB |
| **milvus** | `milvusdb/milvus:v2.5.4` standalone | hybrid vector DB | 2 GB |
| **embedder** | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.6` | `intfloat/multilingual-e5-small` (384-d, RU/EN) | 1 GB |
| **api** | `python:3.11-slim` build | FastAPI RAG | 512 MB |

Не включены: OpenSearch, Dashboards, Postgres/pgvector, Langfuse, Attu, Infinity+BGE-M3.

> Официальный Milvus standalone нельзя свести к одному контейнеру — нужны etcd + MinIO. Это минимальный поддерживаемый стек.

Опционально тяжёлый embedder:

```bash
docker compose --profile heavy up -d
```

## Быстрый старт

```bash
cd F:\Asias\rag-milvus
copy .env.example .env
# задайте OPENAI_API_KEY для /v1/ask

docker compose up -d --build
# API:             http://localhost:8010
# Milvus gRPC:     19530
# TEI embeddings:  8080
```

```bash
curl http://localhost:8010/health
```

Ingest + search:

```bash
curl -X POST http://localhost:8010/v1/admin/ingest/text \
  -H "X-Admin-Key: dev-admin-key" \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"Тест\",\"text\":\"Свято-Успенский монастырь в Красноярске...\",\"org_id\":\"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa\",\"acl\":[\"public\"],\"sync\":true}"

curl -X POST http://localhost:8010/v1/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Что такое Свято-Успенский монастырь?\",\"org_id\":\"*\",\"limit\":5}"
```

`org_id: "*"` — поиск по всем tenants.

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | статус + milvus |
| POST | `/v1/search` | hybrid retrieval |
| POST | `/v1/ask` | streaming RAG ответ |
| POST | `/v1/admin/ingest/text` | sync/async текст → Milvus |
| POST | `/v1/admin/ingest/file` | sync/async файл → Milvus |
| GET | `/v1/admin/ingest/tasks/{id}` | статус фона |

## Локально без Docker API

Нужны уже запущенные Milvus + TEI:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn src.main:app --reload --port 8010
```

## Отличия от rag-platform

- Один движок: Milvus hybrid (dense COSINE + BM25 russian + RRF)
- Нет dual-write / ETL / pgvector / OpenSearch
- Лёгкие эмбеддинги e5-small (384) вместо BGE-M3 (1024)
- Reranker выключен по умолчанию (`RERANKER_URL=`)
