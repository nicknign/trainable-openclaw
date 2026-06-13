"""Test vllm multi-message API — upload and run on remote."""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="no-key")

r = client.chat.completions.create(
    model="qwen3-4b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Reply in 1 sentence."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2=4"},
        {"role": "user", "content": "Now multiply that by 3"},
    ],
    max_tokens=200,
    temperature=0.7,
)
print("Response:", r.choices[0].message.content[:300])
print("Finish:", r.choices[0].finish_reason)
print("Usage:", r.usage)
print("MULTI-MESSAGE API: OK")
