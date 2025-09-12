from redis.asyncio import Redis

from .config import settings


class RedisConfig:
    def __init__(self):
        self.host = settings.redis_host
        self.port = settings.redis_port
        self.db = settings.redis_db

    async def get_redis_connection(self) -> Redis:
        client = Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True
        )
        await client.ping()
        return client
