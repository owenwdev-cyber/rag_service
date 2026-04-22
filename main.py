import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import project modules and configurations
from src.rag_service.core.settings import settings
from src.rag_service.api.routes import router
from src.rag_service.rag_container import rag_service

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application lifespan handler for startup and shutdown events
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Building knowledge base...")
    rag_service.auto_build_kb()
    logger.info("RAG service started successfully")
    yield
    logger.info("Shutting down service")

# Create FastAPI application instance
app = FastAPI(
    title="RAG Intelligent Q&A System",
    description="Document-based Retrieval Augmented Generation Q&A System",
    version="1.0.0",
    lifespan=lifespan  
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes with prefix
app.include_router(router, prefix="/api")

# Run the server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server: http://{settings.server.host}:{settings.server.port}")
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="info"
    )