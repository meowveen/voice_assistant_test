"""
Local voice assistant with SELF-HOSTED web search (SearXNG):
  whisper.cpp (STT) -> Qwen via Ollama + SearXNG tool (LLM) -> Kokoro (TTS)
Turn-based loop: press Enter, speak, hear the reply. The model decides on its
own when a question needs a web search; searches go through your local SearXNG.

Prereqs (see setup instructions):
  - Ollama running with a tool-capable model:  ollama pull qwen3:8b
  - SearXNG container running with JSON enabled at SEARXNG_URL (see settings.yml)
  - whisper.cpp built + large-v3-turbo model downloaded
  - pip install kokoro soundfile sounddevice requests numpy ; brew install espeak-ng

Run inside your Python 3.12 venv:  python voice_assistant.py
"""

import re
import subprocess
import time
from datetime import datetime

import requests
import sounddevice as sd
import soundfile as sf
from kokoro import KPipeline

# --- Config --------------------------------------------------------------
WHISPER_BIN   = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "./whisper.cpp/models/ggml-large-v3-turbo.bin"
OLLAMA_URL    = "http://localhost:11434/api/chat"
MODEL         = "qwen3:8b"        # swap to "qwen3:14b" if you have the RAM
VOICE         = "af_heart"        # try af_bella, am_adam, etc.
RECORD_SECS   = 6                 # crude fixed window; see note about VAD below
SEARXNG_URL   = "http://localhost:8080/search"   # your local SearXNG instance
NUM_RESULTS   = 5                 # how many search hits to feed the model
MAX_TOOL_HOPS = 3                 # cap search rounds so a turn can't loop forever

# Tool schema advertised to the model (Ollama /api/chat function-calling format)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current or factual information "
                       "the assistant is unsure of (news, prices, events, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}]

tts = KPipeline(lang_code="a")    # 'a' = American English

def _ts():
    return datetime.now().strftime("%H:%M:%S")

# --- The search tool (local SearXNG) ------------------------------------
def web_search(query):
    """Query the local SearXNG instance and return the top results."""
    resp = requests.get(
        SEARXNG_URL,
        params={"q": query, "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in resp.json().get("results", [])[:NUM_RESULTS]
    ]
    return {"results": results}

# --- Pipeline stages -----------------------------------------------------
def record(path="input.wav", seconds=RECORD_SECS, sr=16000):
    """Record mono 16 kHz 16-bit WAV (the only format whisper-cli accepts)."""
    print(f"[{_ts()}] Listening for {seconds}s...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    sf.write(path, audio, sr, subtype="PCM_16")

def transcribe(path="input.wav"):
    """Run whisper.cpp on the WAV; -nt strips timestamps for clean text."""
    out = subprocess.run(
        [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", path, "-nt"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()

def chat(text, history):
    """Send the turn to Ollama with the web_search tool available, running a
    call->search->answer loop. think=False keeps qwen3 out of reasoning mode so
    we don't speak its scratch work; the regex strip is a belt-and-suspenders."""
    history.append({"role": "user", "content": text})

    msg = {}
    for _ in range(MAX_TOOL_HOPS):
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": history,
            "tools": TOOLS,
            "stream": False,
            "think": False,
            "options": {"num_ctx": 32000},   # search results are token-heavy
        })
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")
        msg = data["message"]
        history.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            break  # model produced a final answer

        for call in tool_calls:
            fn = call.get("function", {})
            if fn.get("name") == "web_search":
                args = fn.get("arguments", {})
                query = args.get("query", "")
                print(f"  [{_ts()}] [searching: {query}]")
                try:
                    results = web_search(query)
                except Exception as e:
                    results = {"error": f"search failed: {e}"}
                history.append({
                    "role": "tool",
                    "content": str(results)[:8000],  # trim to protect context
                })

    reply = re.sub(r"<think>.*?</think>", "", msg.get("content", "") or "",
                   flags=re.DOTALL).strip()
    history.append({"role": "assistant", "content": reply})
    return reply

def speak(text):
    """Kokoro yields 24 kHz audio chunks; play each as it's generated."""
    # Remove Markdown asterisks (bold/italic), emojis, and stray '*' so Kokoro doesn't speak them
    def _clean_for_tts(s: str) -> str:
        # Unwrap bold/italic markers: **bold** -> bold, *italic* -> italic
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        s = re.sub(r"\*(.+?)\*", r"\1", s)
        # Remove any remaining standalone asterisks
        s = s.replace("*", "")
        # Strip emoji (Unicode ranges + common extended emojis)
        s = re.sub(
            r"[\U0001F600-\U0001F64F"  # Emoticons
            r"\U0001F300-\U0001F5FF"   # Symbols & pictographs
            r"\U0001F680-\U0001F6FF"   # Transport & map symbols
            r"\U0001F1E0-\U0001F1FF"   # Flags (iOS)
            r"\U00002500-\U00002BEF"   # Chinese characters
            r"\U00002702-\U000027B0"   # Dingbats
            r"\U00002702-\U000027B0"   # Other symbols
            r"\U000024C2-\U0001F251"   # Enclosed characters
            r"\U0001f926-\U0001f937"   # Hand gestures
            r"\U00010000-\U0010ffff"   # Other emojis
            r"\u2640-\u2642"           # Gender symbols
            r"\u2600-\u2B55"           # Misc symbols and pictographs
            r"\u200d"                  # Zero width joiner
            r"\u23cf"                  # ⏏
            r"\u23e9"                  # ⏩
            r"\u231a"                  # ⌚
            r"\ufe0f"                  # Dingbats
            r"\u3030"                  # Wavy dash
            "]+",
            "",
            s
        )
        return s.strip()

    text = _clean_for_tts(text)
    for _, _, audio in tts(text, voice=VOICE):
        sd.play(audio, 24000)
        sd.wait()

# --- Main loop -----------------------------------------------------------
def main():
    history = [{
        "role": "system",
        "content": "You are a concise, friendly office concierge of Accenture Singapore. "
                   "Your office is located in Singapore, in the building Raffles City Tower. "
                   "For Singapore generic queries, you must search the internet for more information. "
                   "Reply in polite spoken sentences. IMPORTANT: Use only normal alphabets, numbers, and basic punctuation. "
                   "Do NOT use asterisks, emojis, special Unicode characters, or formatting symbols.",
    }]
    print("Voice assistant ready. Ctrl+C to quit.")
    while True:
        input("\nPress Enter to speak...")
        record()
        t_start = time.monotonic()
        print(f"[{_ts()}] Transcribing...")
        user = transcribe()
        if not user:
            print("(heard nothing)")
            continue
        print(f"[{_ts()}] You: {user}")
        print(f"[{_ts()}] Thinking...")
        reply = chat(user, history)
        elapsed = time.monotonic() - t_start
        print(f"[{_ts()}] Assistant: {reply}  (total response time: {elapsed:.1f}s)")
        print(f"[{_ts()}] Speaking...")
        speak(reply)
        print(f"[{_ts()}] Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
