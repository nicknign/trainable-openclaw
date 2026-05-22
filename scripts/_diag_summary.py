#!/usr/bin/env python3
"""Save diagnostic summary to file to avoid encoding issues."""
import json

with open("/tmp/chat_full.json") as f:
    d = json.load(f)

c = d["choices"][0]["message"]["content"]
ascii_count = sum(1 for ch in c if ord(ch) < 128)
total = len(c)

lines = []
lines.append(f"content_len={total}")
lines.append(f"ascii_count={ascii_count}")
lines.append(f"ascii_ratio={ascii_count/total*100:.1f}%")
lines.append(f"has_Hello={'Hello' in c}")
lines.append(f"has_I={'I am' in c or 'Im' in c}")
lines.append(f"first_50_repr={repr(c[:50])}")
lines.append(f"first_20_hex={c[:20].encode('utf-8').hex()}")

with open("/tmp/chat_diag.txt", "w") as f:
    f.write("\n".join(lines))

print("Diagnostic saved to /tmp/chat_diag.txt")
