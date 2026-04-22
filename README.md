# RAG Service

A production-ready localized RAG (Retrieval-Augmented Generation) Q&A service, providing APIs based on FastAPI, supporting document parsing, vector retrieval, BM25 hybrid recall, cross-encoder reranking, multi-turn dialogue context, and SSE streaming output.

## Project Overview

This project is designed to build an enterprise knowledge Q&A backend service that is "controllable, traceable, and deployable offline", with core objectives:

- Private document indexing and retrieval-augmented Q&A (RAG)
- Mixed processing of Chinese and English documents
- Incrementally updatable knowledge base indexing
- Streaming Q&A interface for frontend conversation scenarios

## Core Features

- Document Parsing: Supports `txt`, `md`, `pdf`, `docx`, `epub`
- Text Chunking: Chinese/English tokenization and chunking
- Vector Retrieval: Semantic recall based on FAISS
- Keyword Retrieval: Lexical recall based on BM25
- Hybrid Retrieval: Vector retrieval + BM25 fusion
- Result Reranking: Cross-Encoder reranking to improve relevance
- Query Rewrite: Question rewriting for multi-turn dialogue (toggleable)
- Doc2Query: Document expansion query enhancement for improved recall (toggleable)
- Session Management: SQLite-persisted sessions and messages
- Streaming Output: SSE (`text/event-stream`) returns incremental tokens

## Technology Stack

- Python 3.10+
- FastAPI + Uvicorn
- FAISS
- sentence-transformers
- rank-bm25
- PyMuPDF / python-docx / EbookLib
- SQLite

## Directory Structure

```text
rag_service/
├─ main.py                         # Application entry point
├─ src/rag_service/
│  ├─ api/routes.py                # HTTP API routes
│  ├─ core/                        # Core retrieval and building logic
│  ├─ models/                      # Embedding / Reranker model wrappers
│  ├─ storage/                     # FAISS and state persistence
│  └─ utils/                       # Document loading and text processing utilities
├─ documents/                      # Document knowledge base directory (runtime)
├─ vector_db/                      # FAISS index (runtime)
├─ cache/                          # Chunk cache (runtime)
├─ records/                        # File hash records (runtime)
├─ tests/                          # Tests and example scripts
├─ .env.example                    # Environment variable template
└─ requirements.txt
```

## Quick Start

### 1) Install Dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure Environment Variables

```bash
cp .env.example .env
```

> Current project default configurations are mainly in `src/rag_service/core/settings.py`; `.env` is used for unified configuration injection during enterprise deployment.

### 3) Prepare Knowledge Documents

Place documents in the following directories (as needed):

- `documents/disable_doc2query/zh`
- `documents/disable_doc2query/en`
- `documents/enable_doc2query/zh`
- `documents/enable_doc2query/en`

### 4) Start the Service

```bash
python main.py
```

Default listening address: `http://0.0.0.0:8000`

Health check:

```bash
GET /api/health
```

## API Documentation

### 1. Create Conversation

`POST /api/conversation/create`

Query parameters:

- `title` (optional): Conversation title

### 2. List Conversations

`GET /api/conversations`

### 3. Delete Conversation

`POST /api/conversation/delete`

Request body:

```json
{
  "conversation_id": "conv_xxx"
}
```

### 4. Paginated Conversation Messages

`GET /api/conversation/{conversation_id}/messages/page?page=1&size=10`

### 5. Delete Message

`POST /api/message/delete`

Request body:

```json
{
  "message_id": "msg_xxx"
}
```

### 6. Streaming Q&A (Core)

`POST /api/chat/stream`

Request body:

```json
{
  "conversation_id": "conv_xxx",
  "question": "Please summarize the key points in the document",
  "top_k": 5
}
```

Response type:

- `text/event-stream`
- Each event contains incremental `text` and cumulative `full`
- Final event includes `type=done`, `message_id`, `conversation_id`

## Operation and Deployment Recommendations (Enterprise-Level)

- Use containerized deployment (basic `Dockerfile` and `docker-compose.yml` provided in the repository)
- Configure independent model cache volumes to avoid repeated model downloads
- Mount persistent storage for `documents/`, `vector_db/`, `cache/`, `records/`
- Treat external model services (e.g., Ollama) as independent observable components
- Unified access through reverse proxy with consolidated CORS, authentication, and rate limiting policies

### Docker Startup

```bash
docker compose up -d --build
```

## Testing

The project currently provides functional test scripts under `tests/` (example/integration verification style):

```bash
pytest -q
```

> For enterprise-level CI, it is recommended to add:
> - Unit test coverage threshold (e.g., 70%+)
> - Type checking (mypy/pyright)
> - Code style checking (ruff/flake8 + black)

## Open Source Governance

This repository has completed common enterprise-level open source basic files:

- `LICENSE`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `.env.example`

## Security Notes

- Do not commit real keys, production addresses, or internal documents
- In production environments, strictly limit CORS sources; do not use `*`
- It is recommended to add authentication (JWT/OAuth2/API Key) and rate limiting policies at the gateway layer

## License

This project is licensed under the [MIT License](./LICENSE).