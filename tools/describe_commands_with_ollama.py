#!/usr/bin/env python3
"""
Generate benign descriptions for Linux commands using a local Ollama model
or OpenAI models. Optionally produce K benign paraphrases ("mutation fuzzing")
per generated description.

Inputs:
- linux-index-with-commands.csv (must include a 'Command' column)
- command.txt (prompt template containing <COMMANDD_HERE> or <COMMAND_HERE>)

Output:
- Writes a new CSV with an added 'Description' column.

Example:
  python tools/describe_commands_with_ollama.py \
    --index linux-index-with-commands.csv \
    --template command.txt \
    --out linux-index-with-descriptions.csv \
    --model llama3:8b
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests
import sys
import os
 
# Verbose flag (set in main from --print-io)
VERBOSE: bool = False

# Optional: OpenAI provider (pip install openai>=1.0.0)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # handled at runtime if provider=openai

# Optional: Google Vertex AI (Gemini) provider (pip install google-cloud-aiplatform>=1.66.0)
try:
    import vertexai  # type: ignore
    from vertexai.generative_models import GenerativeModel, GenerationConfig  # type: ignore
except Exception:
    vertexai = None  # type: ignore
    GenerativeModel = None  # type: ignore
    GenerationConfig = None  # type: ignore


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, command: str) -> str:
    # Support either placeholder spelling
    if "<COMMANDD_HERE>" in template:
        return template.replace("<COMMANDD_HERE>", command)
    if "<COMMAND_HERE>" in template:
        return template.replace("<COMMAND_HERE>", command)
    # Fallback: append the command at the end in a minimal format
    return f"{template.rstrip()}\n\nInput Command: {command}\nOutput:"


MUTATION_PROMPT = """You are a precise paraphraser. Given a benign task sentence, generate {K} distinct, SHORT paraphrases that preserve benign meaning and constraints.
Rules:
- ≤ 25 words each
- No security/adversarial jargon
- Output JSON array of strings ONLY (no commentary).

Sentence: "{SENTENCE}"
"""


def call_ollama_generate(
    model: str,
    prompt: str,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.2,
    top_p: float = 0.9,
    retries: int = 3,
    timeout_seconds: float = 60.0,
) -> Optional[str]:
    url = f"{host.rstrip('/')}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, data=json.dumps(payload), timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            # Non-streaming returns a single object with 'response'
            text = data.get("response", "")
            # Normalize to one line and trim whitespace
            return " ".join(text.strip().split())
        except Exception as e:
            last_error = e
            if VERBOSE:
                print(f"[ollama-error attempt={attempt+1}] {e}")
            # brief backoff
            time.sleep(0.75 * (attempt + 1))
    return None


def call_ollama_generate_stream(
    model: str,
    prompt: str,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout_seconds: float = 60.0,
    on_chunk: Optional[Any] = None,
) -> Optional[str]:
    """
    Stream tokens from Ollama generate API. Accumulates and returns the final text.
    If on_chunk is provided, it will be called with each text delta.
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    try:
        with requests.post(url, data=json.dumps(payload), stream=True, timeout=timeout_seconds) as resp:
            resp.raise_for_status()
            full_text_parts: list[str] = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = data.get("response", "")
                if delta:
                    full_text_parts.append(delta)
                    if on_chunk:
                        on_chunk(delta)
                if data.get("done"):
                    break
            text = "".join(full_text_parts)
            return " ".join(text.strip().split())
    except Exception as e:
        if VERBOSE:
            print(f"[ollama-stream-error] {e}")
        return None


def call_openai_generate(
    model: str,
    prompt: str,
    stream: bool = False,
    temperature: float = 1,
    max_tokens: int = 200,
) -> Optional[str]:
    """
    Generate text using OpenAI chat completions (requires OPENAI_API_KEY).
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai>=1.0.0")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    # Configure a client-level timeout to avoid indefinite hangs
    try:
        client_timeout = float(os.environ.get("OPENAI_TIMEOUT", "60"))
    except Exception:
        client_timeout = 60.0
    client = OpenAI(timeout=client_timeout)
    # Use Responses API for GPT-5 family; Chat Completions for others
    m = (model or "").lower()
    is_gpt5 = ("gpt-5-min" in m) or ("gpt-5-mini" in m) or m.startswith("gpt-5")
    if is_gpt5:
        try:
            # Prefer non-stream for GPT-5 for reliability
            resp = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max(512, max_tokens),
                temperature=1,  # GPT-5 requires default temperature
                reasoning={"effort": "low"},
            )
            text = getattr(resp, "output_text", "") or ""
            if not text:
                # Fallback extraction for SDKs without output_text helper
                try:
                    parts: List[str] = []
                    outs = getattr(resp, "output", []) or []
                    for o in outs:
                        content = getattr(o, "content", []) or []
                        for c in content:
                            t = getattr(c, "text", None)
                            if t is not None:
                                val = getattr(t, "value", None)
                                if isinstance(val, str):
                                    parts.append(val)
                    text = " ".join(parts)
                except Exception:
                    pass
            return " ".join((text or "").strip().split())
        except Exception as e:
            if VERBOSE:
                print(f"[openai-error] {e}")
            return None
    # Non-GPT-5 models use Chat Completions
    if not stream:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            return " ".join(text.split())
        except Exception as e:
            if VERBOSE:
                print(f"[openai-error] {e}")
            return None
    try:
        full: List[str] = []
        with client.chat.completions.with_streaming_response.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ) as stream_resp:
            for event in stream_resp:
                if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                    chunk = event.choices[0].delta.content
                    full.append(chunk)
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            sys.stdout.write("\n")
        text = "".join(full).strip()
        return " ".join(text.split())
    except Exception as e:
        if VERBOSE:
            print(f"[openai-stream-error] {e}")
        return None


def call_gemini_generate(
    model: str,
    prompt: str,
    project: Optional[str],
    location: Optional[str],
    stream: bool = False,
    temperature: float =0.2,
    max_tokens: int = 200,
) -> Optional[str]:
    """
    Generate text using Google Vertex AI Gemini models.
    Requires authentication (e.g., gcloud auth application-default login)
    and project/location provided via args or env:
      GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
    """
    if vertexai is None or GenerativeModel is None:
        raise RuntimeError("Vertex AI SDK not available. pip install google-cloud-aiplatform")
    # Fall back to env if args not provided
    proj = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    loc = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
    if not proj:
        raise RuntimeError("Gemini requires a project. Set --gemini-project or GOOGLE_CLOUD_PROJECT.")
    try:
        vertexai.init(project=proj, location=loc)
        gen = GenerativeModel(model)
        gen_cfg = GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
        if not stream:
            resp = gen.generate_content(prompt, generation_config=gen_cfg)
            text = getattr(resp, "text", "") or ""
            return " ".join(text.strip().split())
        # streaming
        full_parts: List[str] = []
        for chunk in gen.generate_content(prompt, generation_config=gen_cfg, stream=True):
            # chunk may expose .text for partials
            piece = getattr(chunk, "text", "") or ""
            if piece:
                full_parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
        sys.stdout.write("\n")
        return " ".join("".join(full_parts).strip().split())
    except Exception as e:
        if VERBOSE:
            print(f"[gemini-error] {e}")
        return None


def mutate_sentence(
    provider: str,
    model: str,
    sentence: str,
    k: int,
    temperature: float,
    max_tokens: int,
    debug: bool = False,
) -> List[str]:
    if k <= 0 or not sentence:
        return []
    p = MUTATION_PROMPT.replace("{K}", str(k)).replace("{SENTENCE}", sentence.replace('"', '\\"'))
    if debug:
        print("=== MUTATIONS: PROMPT ===")
        print(p)
        print("=========================")
    if provider == "ollama":
        raw = call_ollama_generate(model=model, prompt=p) or ""
    elif provider == "gemini":
        raw = call_gemini_generate(
            model=model,
            prompt=p,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
            stream=False,
            temperature=max(0.4, temperature),
            max_tokens=max_tokens,
        ) or ""
    else:
        raw = call_openai_generate(model=model, prompt=p, stream=False, temperature=max(0.4, temperature), max_tokens=max_tokens) or ""
    if debug:
        print("=== MUTATIONS: OUTPUT RAW ===")
        print(raw)
        print("=============================")
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    # Fallback naive split by lines
    return [x.strip("- ").strip() for x in raw.splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate benign descriptions for commands using Ollama, OpenAI, or Gemini, with optional mutation fuzzing.")
    ap.add_argument("--index", default="linux-index-with-commands.csv", help="Path to input CSV (must include 'Command').")
    ap.add_argument("--template", default="command.txt", help="Path to the LLM prompt template.")
    ap.add_argument("--out", default="linux-index-with-descriptions.csv", help="Output CSV path.")
    ap.add_argument("--provider", choices=["ollama", "openai", "gemini"], default="ollama", help="LLM provider.")
    ap.add_argument("--model", default="llama3:8b", help="Model name (e.g., llama3:8b, gpt-4o-mini, gemini-1.5-flash-001).")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama host URL.")
    ap.add_argument("--temperature", type=float, default=1, help="Sampling temperature.")
    ap.add_argument("--max-tokens", type=int, default=200, help="Max tokens for OpenAI generations.")
    ap.add_argument("--mutations", type=int, default=0, help="If >0, generate this many benign paraphrases per description.")
    ap.add_argument("--skip-empty", action="store_true", help="Skip rows with empty Command instead of writing blank Description.")
    ap.add_argument("--stream", action="store_true", help="Stream tokens to stdout while generating descriptions (if supported).")
    ap.add_argument("--print-io", action="store_true", help="Print the full prompt and model output for each row.")
    # Gemini-specific
    ap.add_argument("--gemini-project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), help="GCP project for Vertex AI (fallback to GOOGLE_CLOUD_PROJECT).")
    ap.add_argument("--gemini-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"), help="Vertex AI location (fallback to GOOGLE_CLOUD_LOCATION).")
    args = ap.parse_args()

    index_path = Path(args.index)
    template_path = Path(args.template)
    out_path = Path(args.out)

    global VERBOSE
    VERBOSE = bool(args.print_io)

    template = read_text(template_path)

    with index_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "Command" not in fieldnames:
        raise SystemExit("Input CSV must include a 'Command' column.")
    if "Description" not in fieldnames:
        fieldnames.append("Description")
    if args.mutations > 0 and "Mutations" not in fieldnames:
        fieldnames.append("Mutations")

    completed = 0
    for i, row in enumerate(rows):
        command = (row.get("Command") or "").strip()
        if not command:
            if args.skip_empty:
                row["Description"] = ""
                continue
            row["Description"] = ""
            continue

        prompt = build_prompt(template, command)
        if args.print_io:
            print(f"\n=== ROW {i+1}/{len(rows)} | {args.provider}:{args.model} ===")
            print("=== PROMPT ===")
            print(prompt)
            print("==============")
        if args.provider == "ollama":
            if args.stream:
                preview = command.replace("\n", " ")[:120]
                print(f"[{completed+1}/{len(rows)}] {preview}")
                if args.print_io:
                    print("=== OUTPUT (stream) ===")
                def _on_chunk(tok: str) -> None:
                    sys.stdout.write(tok)
                    sys.stdout.flush()
                desc = call_ollama_generate_stream(
                    model=args.model,
                    prompt=prompt,
                    host=args.ollama_host,
                    temperature=args.temperature,
                    on_chunk=_on_chunk,
                )
                print("")
            else:
                desc = call_ollama_generate(
                    model=args.model,
                    prompt=prompt,
                    host=args.ollama_host,
                    temperature=args.temperature,
                )
                if args.print_io:
                    print("=== OUTPUT ===")
                    print(desc or "")
                    print("==============")
        else:
            # OpenAI
            if args.provider == "openai":
                desc = call_openai_generate(
                    model=args.model,
                    prompt=prompt,
                    stream=args.stream,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                if args.print_io and not args.stream:
                    print("=== OUTPUT ===")
                    print(desc or "")
                    print("==============")
                if args.print_io and args.stream:
                    # Streaming is printed by the SDK loop already; add final normalized echo
                    print("\n=== OUTPUT (final) ===")
                    print((desc or "").strip())
                    print("======================")
            else:
                # Gemini via Vertex AI
                desc = call_gemini_generate(
                    model=args.model,
                    prompt=prompt,
                    project=args.gemini_project,
                    location=args.gemini_location,
                    stream=args.stream,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                if args.print_io and not args.stream:
                    print("=== OUTPUT ===")
                    print(desc or "")
                    print("==============")
                if args.print_io and args.stream:
                    print("\n=== OUTPUT (final) ===")
                    print((desc or "").strip())
                    print("======================")
        row["Description"] = desc or ""

        if args.mutations > 0 and row["Description"]:
            muts = mutate_sentence(
                provider=args.provider,
                model=args.model,
                sentence=row["Description"],
                k=args.mutations,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                debug=bool(args.print_io),
            )
            row["Mutations"] = json.dumps(muts, ensure_ascii=False)

        completed += 1

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path} (generated descriptions for {completed} commands).")


if __name__ == "__main__":
    main()

