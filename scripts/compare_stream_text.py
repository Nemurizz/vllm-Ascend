#!/usr/bin/env python3
"""Save one deterministic streaming response for A/B correctness comparison."""

import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    body = json.dumps({
        "model": "Qwen3-8B",
        "messages": [{"role": "user", "content": "请用一句话解释缓存。"}],
        "max_tokens": 32,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
    }).encode()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    parts = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            for choice in event.get("choices", []):
                parts.append(choice.get("delta", {}).get("content") or "")
    text = "".join(parts)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"text": text, "text_chars": len(text)}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"text": text, "text_chars": len(text)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
