#!/usr/bin/env python3
"""Interactive chat CLI for the veRL inference server.

Usage:
    python scripts/chat.py                          # default http://localhost:8000
    python scripts/chat.py http://remote:8000       # custom server
    python scripts/chat.py -t 0.5 -m 512            # temperature + max_tokens
    python scripts/chat.py -M gpt-4o                # model name for load balancing
"""

import json
import os
import sys
import textwrap
from urllib.error import URLError
from urllib.request import Request, urlopen


def chat_loop(
    base_url: str = "http://localhost:8000",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    thinking: bool = True,
    model: str = "default",
):
    """Interactive multi-turn chat loop."""
    url = f"{base_url}/v1/chat/completions"
    messages: list[dict] = []

    print(f"\n{'='*70}")
    print(f"  veRL Chat CLI")
    print(f"  Server:    {base_url}")
    print(f"  Temperature: {temperature}  Max Tokens: {max_tokens}  Thinking: {'ON' if thinking else 'OFF'}")
    print(f"  Commands:  /clear  /undo  /think  /quit")
    print(f"{'='*70}\n")

    turn = 0

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n/quit")
            break

        if not user_input:
            continue

        # ---- Commands ----
        if user_input == "/quit" or user_input == "/exit":
            print("Bye.")
            break
        if user_input == "/clear":
            messages.clear()
            turn = 0
            print("[history cleared]\n")
            continue
        if user_input == "/undo":
            if messages:
                removed = messages.pop()
                if messages:
                    # Remove the last user message too
                    if messages[-1].get("role") != "user":
                        messages.pop()
                turn -= 1
                print(f"[removed last turn: {removed['role']}: {removed['content'][:50]}...]\n")
            else:
                print("[nothing to undo]\n")
            continue
        if user_input == "/think":
            thinking = not thinking
            print(f"[thinking: {'ON' if thinking else 'OFF'}]\n")
            continue

        # ---- Normal chat ----
        turn += 1
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_thinking": thinking,
        }

        # Show full request
        print(f"\n{'─'*60}")
        print(f"┌─ REQUEST (turn {turn})")
        print(f"│  URL: {url}")
        print(f"│  Payload:")
        for line in json.dumps(payload, ensure_ascii=False, indent=4).split("\n"):
            print(f"│  {line}")
        print(f"{'─'*60}")

        try:
            req = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=300)
            data = json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            print(f"[ERROR] Cannot reach server: {e}")
            messages.pop()
            turn -= 1
            continue
        except Exception as e:
            print(f"[ERROR] {e}")
            messages.pop()
            turn -= 1
            continue

        # Show full response
        print(f"┌─ RESPONSE (turn {turn})")
        for line in json.dumps(data, ensure_ascii=False, indent=4).split("\n"):
            print(f"│  {line}")
        print(f"{'─'*60}")

        # Extract assistant reply and append to history
        try:
            reply = data["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": reply})
            print(f"\nAssistant>\n{reply}\n")
        except (KeyError, IndexError) as e:
            print(f"\n[ERROR] Malformed response: {e}\n")
            messages.pop()
            turn -= 1


def main():
    args = sys.argv[1:]
    base_url = "http://localhost:8000"
    temperature = 0.7
    max_tokens = 2048
    thinking = True
    model = "default"

    i = 0
    while i < len(args):
        if args[i] == "-h" or args[i] == "--help":
            print(__doc__)
            sys.exit(0)
        elif args[i] == "-t" and i + 1 < len(args):
            temperature = float(args[i + 1])
            i += 2
        elif args[i] == "-m" and i + 1 < len(args):
            max_tokens = int(args[i + 1])
            i += 2
        elif args[i] == "-M" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--no-think":
            thinking = False
            i += 1
        elif args[i].startswith("-"):
            print(f"Unknown option: {args[i]}")
            sys.exit(1)
        else:
            base_url = args[i].rstrip("/")
            i += 1

    chat_loop(base_url, temperature=temperature, max_tokens=max_tokens, thinking=thinking, model=model)


if __name__ == "__main__":
    main()
