import time, json
from openai import OpenAI

t0 = time.time()
client = OpenAI(
    api_key="sk-906ad0dc48354e7aba594ef6d9aa5be6",
    base_url="https://api.deepseek.com",
)

# Test 1: basic
print("=== Test 1: basic ===")
r = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Say exactly: hello"}],
    max_tokens=50,
    temperature=0.0,
    extra_body={"thinking": {"type": "disabled"}},
)
print(f"Content: '{r.choices[0].message.content}'")
print(f"Finish: {r.choices[0].finish_reason}")
print(f"Usage: {r.usage}")
print(f"Time: {time.time()-t0:.2f}s")

# Test 2: without extra_body
print()
print("=== Test 2: without thinking disabled ===")
t0 = time.time()
r = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Say exactly: hello"}],
    max_tokens=50,
    temperature=0.0,
)
print(f"Content: '{r.choices[0].message.content}'")
print(f"Finish: {r.choices[0].finish_reason}")
print(f"Time: {time.time()-t0:.2f}s")
