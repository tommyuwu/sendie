from typing import List, Dict, Any


def build_messages(
        system_prompt: str,
        history: List[Dict[str, str]],
        knowledge: List[str],
        user_message: str
) -> str:
    messages = [{
        "role": "system",
        "content": system_prompt
    }]

    if knowledge:
        knowledge_context = "INFORMACIÓN DISPONIBLE DEL SISTEMA:\n\n"
        for i, doc in enumerate(knowledge, 1):
            knowledge_context += f"[Documento {i}]\n{doc}\n\n"

        messages.append({
            "role": "system",
            "content": knowledge_context + "\nUsa esta información para responder las consultas del usuario de manera precisa.\n\n"
        })

    if history and len(history) > 0:
        messages.append({
            "role": "system",
            "content": "HISTORIAL DE MENSAJES:\n"
        })
        for msg in history[-6:]:
            messages.append(msg)
    else:
        messages.append({
            "role": "system",
            "content": "**NUEVA CONVERSACION**\n"
        })

    messages.append({
        "role": "user",
        "content": user_message
    })

    return flatten_messages(messages)


def flatten_messages(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        lines.append(f"{m['role'].capitalize()}: {m['content']}")
    return "\n".join(lines)
