#!/usr/bin/env python3
"""GPT-5.4 client for a limited-scope comparison arm.

Wraps the Azure-NAIRR GPT-5.4 OpenAI-compatible endpoint (messi.eecs.umich.edu) behind the SAME
`.call(jpegs, prompt) -> str` interface as baseline_qualcomm_zeroshot.QwenText, so it drops into
the existing turn-based / gated baseline unchanged. The api_key is read from the repo `.env`
field `api_key` (never hard-coded).

Quick check:  python eval/gpt_client.py            # text ping
              python eval/gpt_client.py --vision    # 1x1 image ping
"""
import os, time, base64


def read_env(key, env_path=None):
    """Read a single field from the repo .env (KEY=VALUE lines). Does not print the value."""
    env_path = env_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    raise KeyError(f"{key} not found in {env_path}")


class GPT54Text:
    """OpenAI-compatible vision client. Mirrors QwenText.call(jpegs, prompt)."""
    BASE_URL = os.getenv("GPT54_BASE_URL", "http://messi.eecs.umich.edu:8080/v1")
    MODEL = os.getenv("GPT54_MODEL", "azure-nairr-gpt-5-4")

    def __init__(self, timeout=120, retries=2, backoff_s=2.0, max_tokens=120, max_completion=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=read_env("api_key"), base_url=self.BASE_URL, timeout=timeout)
        self.model = self.MODEL
        self.retries, self.backoff_s = retries, backoff_s
        self.max_tokens = max_tokens

    def call(self, jpegs, prompt):
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                   for j in jpegs]
        content.append({"type": "text", "text": prompt})
        last = None
        for attempt in range(self.retries + 1):
            try:
                kw = {"model": self.model,
                      "messages": [{"role": "user", "content": content}]}
                # reasoning models (gpt-5.x) use max_completion_tokens and reject temperature/max_tokens
                kw["max_completion_tokens"] = max(self.max_tokens, 2048)
                r = self.client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - mirror QwenText retry behaviour
                last = e
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise last


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true")
    a = ap.parse_args()
    c = GPT54Text()
    if a.vision:
        # 1x1 white JPEG
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, "JPEG")
        print(repr(c.call([buf.getvalue()], "Reply with the single word OK.")))
    else:
        print(repr(c.call([], "Reply with the single word OK.")))
