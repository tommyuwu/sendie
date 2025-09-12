from fastapi import APIRouter, Request, Depends
from redis.asyncio import Redis

router = APIRouter()


async def get_redis_client(request: Request) -> Redis:
    return request.app.state.redis


@router.get("/redis")
async def get_redis(redis: Redis = Depends(get_redis_client)):
    pong = await redis.ping()
    return {"status": "Redis connected", "pong": pong}
