from ollama import chat

# -------------------------------
# LLM CALL
# -------------------------------
def call_llm(prompt):
    response = chat(
        model='qwen3',
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.message.content

call_llm("hello")