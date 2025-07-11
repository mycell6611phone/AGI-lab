import requests

def call_llm(prompt, system_msg=None, model="llama3", host="http://localhost:11434"):
    """
    Call a local Llama 3 API (Ollama) with the given prompt and system message.
    Returns the response text.
    """
    # Compose system/user prompt (Ollama doesn't have roles, so prepend system message)
    content = ""
    if system_msg:
        content += f"[SYSTEM]: {system_msg}\n"
    content += f"[USER]: {prompt}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "options": { "num_ctx": 4096}
    }

    url = f"{host}/api/chat"
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    # Ollama returns { 'message': ... }
    msg = data.get("message", "")
    if isinstance(msg, dict):
        return msg.get("content", "").strip()
    return str(msg).strip()

