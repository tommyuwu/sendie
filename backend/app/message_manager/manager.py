from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from redis.asyncio import Redis
from ..api.models import ChatRequest
from ..core.config import settings


class MessageManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

    def is_similar(self, new_msg: str, old_msg: str) -> bool:
        v1 = self.embeddings.embed_query(new_msg)
        v2 = self.embeddings.embed_query(old_msg)
        sim = cosine_similarity([v1], [v2])[0][0]
        return sim > 0.9

    async def get_history(self, session_id: str, redis: Redis) -> list:
        history_key = f"chat:history:{session_id}"
        raw_history = await redis.lrange(history_key, -10, -1)
        history = []
        for line in raw_history:
            line_decoded = line.decode('utf-8') if isinstance(line, bytes) else line
            if line_decoded.startswith("Usuario:"):
                history.append({"role": "user", "content": line_decoded[8:].strip()})
            elif line_decoded.startswith("Bot:"):
                history.append({"role": "assistant", "content": line_decoded[4:].strip() + "\n"})

        return history

    async def push_to_history(self, req: ChatRequest, response: str, redis: Redis) -> list:
        trivial = {"hola", "gracias", "ok", "buenas", "chau", "adiós", "adios"}
        history_key = f"chat:history:{req.session_id}"
        if len(req.message.strip()) > 3 and req.message.lower() not in trivial:
            await redis.rpush(history_key, f"Usuario: {req.message}")
        await redis.rpush(history_key, f"Bot: {response}")
        await redis.ltrim(history_key, -10, -1)
        await redis.expire(history_key, 3600)
