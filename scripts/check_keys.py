"""Check that the keys in .env work, without printing them.

Lists the Gemini models this key can call, makes one tiny JSON call, and fetches one company's
XBRL facts from the SEC with the declared User-Agent.
"""
import os
import sys

import requests

from tieout.llm import load_env


def main():
    load_env()
    names = [k for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "SEC_USER_AGENT", "TIEOUT_LLM") if os.environ.get(k)]
    print("set in environment/.env:", names)
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                         headers={"x-goog-api-key": key}, timeout=60)
        print("gemini list models: HTTP", r.status_code)
        if r.ok:
            models = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", [])]
            print("  generateContent models:", ", ".join(sorted(models)))
        else:
            print("  ", r.text[:300])
    groq = os.environ.get("GROQ_API_KEY")
    if groq:
        r = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {groq}"}, timeout=60)
        print("groq list models: HTTP", r.status_code)
        if r.ok:
            print("  models:", ", ".join(sorted(m["id"] for m in r.json().get("data", []))))
    ua = os.environ.get("SEC_USER_AGENT")
    if ua:
        r = requests.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
                         headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=120)
        print("sec companyfacts (Apple): HTTP", r.status_code, "bytes", len(r.content))
        if r.ok:
            facts = r.json().get("facts", {})
            print("  taxonomies:", list(facts), "| us-gaap tags:", len(facts.get("us-gaap", {})))
        print("  user agent looks like 'Name email':", "@" in ua and " " in ua.strip())
    if len(sys.argv) > 1 and key:
        model = sys.argv[1]
        body = {"contents": [{"role": "user", "parts": [{"text": 'Reply with the JSON {"ok": true} and nothing else.'}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          headers={"x-goog-api-key": key}, json=body, timeout=120)
        print(f"gemini call {model}: HTTP", r.status_code, (r.json()["candidates"][0]["content"]["parts"][0]["text"][:60]
                                                             if r.ok else r.text[:300]))


if __name__ == "__main__":
    main()
