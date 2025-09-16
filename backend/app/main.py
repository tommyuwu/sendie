from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.api import router as api_router
from .api.chats import router as chat_router
from .core.config import settings
from .core.logging import configure_logging
from .core.redis_conf import RedisConfig

redis_config = RedisConfig()
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = await redis_config.get_redis_connection()
    app.state.redis = redis
    yield
    await redis.close()
    await redis.connection_pool.disconnect()


app = FastAPI(title=settings.title, version="1.0.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": settings.title,
        "version": "1.0.1",
        "provider": settings.model_provider.value
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
