import json
import os
import logging
import uuid
import sqlite3
import requests
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag_service.core.settings import settings
from src.rag_service.rag_container import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "rag_chat.db")


def success_response(data=None):
    return {
        "code": 200,
        "msg": "success",
        "data": data if data is not None else {}
    }


def error_response(msg: str, code: int = 500):
    return {
        "code": code,
        "msg": msg,
        "data": None
    }


def get_db_connection():
    """Create and return a new SQLite database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database_tables():
    """Initialize database tables automatically."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id VARCHAR(64) NOT NULL UNIQUE,
        title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id VARCHAR(64) NOT NULL,
        message_id VARCHAR(64) NOT NULL UNIQUE,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Database tables initialized successfully")


init_database_tables()


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    top_k: Optional[int] = None


class MessageDeleteRequest(BaseModel):
    message_id: str


class ConversationDeleteRequest(BaseModel):
    conversation_id: str


@router.post("/conversation/create")
def create_conversation(title: str = "New Conversation"):
    """Create a new independent conversation group."""
    conversation_id = f"conv_{uuid.uuid4().hex[:16]}"
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (conversation_id, title) VALUES (?, ?)",
            (conversation_id, title)
        )
        conn.commit()
        cur.close()
        conn.close()
        return success_response({
            "conversation_id": conversation_id,
            "title": title
        })
    except Exception as e:
        logger.error(f"Create conversation error: {e}")
        raise HTTPException(status_code=500, detail=error_response("Failed to create conversation"))


@router.get("/conversations")
def get_conversations():
    """Get all conversation groups for the left sidebar list."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM conversations ORDER BY create_time DESC")
        convs = [dict(row) for row in cur.fetchall()]
        cur.close()
        conn.close()
        return success_response({"conversations": convs})
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return error_response("Failed to get conversations")


@router.post("/conversation/delete")
def delete_conversation(req: ConversationDeleteRequest):
    """Delete an entire conversation and all its messages."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE conversation_id = ?", (req.conversation_id,))
        cur.execute("DELETE FROM conversations WHERE conversation_id = ?", (req.conversation_id,))
        conn.commit()
        cur.close()
        conn.close()
        return success_response({"success": True})
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=error_response("Delete conversation failed"))


@router.get("/conversation/{conversation_id}/messages/page")
def get_messages_page(conversation_id: str, page: int = 1, size: int = 10):
    """Get paginated messages for a specified conversation."""
    try:
        offset = (page - 1) * size

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id, message_id, conversation_id, question, answer, create_time
            FROM messages
            WHERE conversation_id = ?
            ORDER BY create_time ASC
            LIMIT ? OFFSET ?
        """, (conversation_id, size, offset))

        messages = [dict(row) for row in cur.fetchall()]

        cur.execute("""
            SELECT COUNT(*) AS total
            FROM messages
            WHERE conversation_id = ?
        """, (conversation_id,))

        total_row = cur.fetchone()
        total = total_row["total"] if total_row else 0

        cur.close()
        conn.close()

        return success_response({
            "messages": messages,
            "page": page,
            "size": size,
            "total": total,
            "pages": (total + size - 1) // size
        })
    except Exception as e:
        logger.error(f"Get paginated messages error: {e}")
        raise HTTPException(status_code=500, detail=error_response("Get paginated messages failed"))


@router.post("/message/delete")
def delete_message(req: MessageDeleteRequest):
    """Delete a single message by message ID."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM messages WHERE message_id = ?",
            (req.message_id,)
        )
        conn.commit()
        cur.close()
        conn.close()
        return success_response({"success": True})
    except Exception as e:
        logger.error(f"Delete message error: {e}")
        raise HTTPException(status_code=500, detail=error_response("Delete message failed"))


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat API with RAG support.
    Automatically loads context from database by conversation_id.
    """
    async def generate_response():
        try:
            user_query = req.question
            conversation_id = req.conversation_id or f"conv_{uuid.uuid4().hex[:16]}"
            top_k = req.top_k or settings.search.default_top_k

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO conversations (conversation_id, title) VALUES (?, ?)",
                    (conversation_id, user_query[:20] + "...")
                )
                conn.commit()
            cur.close()
            conn.close()

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT question, answer FROM messages WHERE conversation_id = ? ORDER BY create_time DESC LIMIT 5",
                (conversation_id,)
            )
            history_rows = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()

            chat_history = list(reversed(history_rows))
            final_query = user_query

            if settings.model.enable_query_rewrite and chat_history:
                history_text = "\n".join([
                    f"用户：{h['question']}\nAI：{h['answer']}" for h in chat_history
                ])

                rewrite_prompt = f"""
请将最后一轮问题改写为适合检索的独立问句。
只输出改写后的结果。

对话历史：
{history_text}

问题：
{user_query}
""".strip()

                try:
                    resp = requests.post(
                        f"{settings.model.ollama_url}/api/generate",
                        json={
                            "model": settings.model.ollama_model,
                            "prompt": rewrite_prompt,
                            "temperature": 0.1,
                            "stream": False
                        },
                        timeout=settings.model.ollama_timeout
                    )
                    final_query = resp.json()["response"].strip()
                except Exception as e:
                    logger.warning(f"Query rewrite failed: {e}")
            _, rag_prompt = rag_service.search(final_query, top_k=top_k)

            try:
                ref_content = rag_prompt.split("参考资料：")[1].split("用户问题：")[0]
            except Exception:
                ref_content = "暂无参考资料"

            history_text = "\n".join([
                f"用户：{h['question']}\nAI：{h['answer']}" for h in chat_history
            ]) if chat_history else ""

            full_prompt = f"""
你是一个专业的问答助手。
请只根据参考资料回答问题。
如果无法找到答案，请回复：“根据现有资料无法回答”。
不要编造信息。

对话历史：
{history_text}

参考资料：
{ref_content}

用户问题：
{user_query}

回答：
""".strip()

            response = requests.post(
                f"{settings.model.ollama_url}/api/generate",
                json={
                    "model": settings.model.ollama_model,
                    "prompt": full_prompt,
                    "temperature": 0.7,
                    "stream": True,
                    "num_predict": 1024
                },
                stream=True,
                timeout=180
            )

            full_answer = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    full_answer += token
                    yield f"data: {json.dumps({'text': token, 'full': full_answer}, ensure_ascii=False)}\n\n"

            message_id = f"msg_{uuid.uuid4().hex[:16]}"
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO messages (conversation_id, message_id, question, answer)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, message_id, user_query, full_answer))
            conn.commit()
            cur.close()
            conn.close()

            yield f"data: {json.dumps({'type': 'done', 'message_id': message_id, 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@router.get("/health")
def health_check():
    """Service health check endpoint."""
    return success_response({
        "status": "healthy",
        "service": "RAG Chat Service"
    })