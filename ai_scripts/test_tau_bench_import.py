"""Quick test of tau_bench tools import."""
import sys
sys.path.insert(0, "/data/wangye/trainable-openclaw")
sys.path.insert(0, "/data/wangye/trainable-openclaw/nanobot-0.2.1")

import nanobot.agent.tools.tau_bench as tb
print("Tools found:")
count = 0
for name in dir(tb):
    obj = getattr(tb, name)
    if isinstance(obj, type) and hasattr(obj, "name") and getattr(obj, "name", ""):
        print(f"  {obj.name}")
        count += 1
print(f"Total tool classes: {count}")
