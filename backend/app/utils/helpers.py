def build_messages(
        system_prompt: str,
        history: list[dict],
        knowledge: list[str],
        user_message: str
) -> str:
    messages = [{"role": "system", "content": system_prompt}]

    messages.extend(history)

    if knowledge:
        docs_text = "\n- ".join(knowledge)
        messages.append({
            "role": "system",
            "content": f"Contexto recuperado de la base de conocimiento:\n- {docs_text}"
        })

    messages.append({"role": "user", "content": user_message})

    return flatten_messages(messages)


def flatten_messages(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        lines.append(f"{m['role'].capitalize()}: {m['content']}")
    return "\n".join(lines)
