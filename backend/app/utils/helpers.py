def create_message(user_input: str, knowledge: list[str], prompt: str) -> str:
    input_parts = [f"Prompt:\n{prompt}"]
    if knowledge:
        docs_text = "\n\n".join(knowledge)
        input_parts.append(f"Knowledge:\n{docs_text}")
    input_parts.append(f"User input: {user_input}")
    return "\n\n".join(input_parts)
