import os, time
from groq import Groq

FAST = "openai/gpt-oss-20b"
BIG = "openai/gpt-oss-120b"   # check current model names in the Groq console
_client = None

def llm(messages, model=BIG, temperature=0, json_mode=False, retries=3):
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    for i in range(retries):
        try:
            r = _client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=900, **kw)
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and i < retries - 1:
                time.sleep(2 ** (i + 1))
                continue
            raise