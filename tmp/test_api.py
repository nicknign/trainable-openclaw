import time
from openai import OpenAI

t0 = time.time()
client = OpenAI(
    api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6",
    base_url="https://api.deepseek.com",
)
r = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Say hello in one word"}],
    max_tokens=10,
    temperature=0.3,
)
print(f"Response: {r.choices[0].message.content}")
print(f"Time: {time.time()-t0:.2f}s")
