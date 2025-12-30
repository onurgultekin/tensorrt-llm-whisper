from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "voiceos-text-formatter"
VOLUME_NAME = "voiceos-text-formatter"
MOUNT_PATH = "/vol"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _image():
    # Keep this separate from `modal_app.py` (Whisper) since TRT‑LLM LLM API uses a different stack
    # (Python 3.12 + TRT‑LLM 0.18) and we don't want to couple deployments.
    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
        .pip_install(
            "tensorrt-llm==0.18.0",
            "pynvml<12",
            "flashinfer-python==0.2.5",
            "cuda-python==12.9.1",
            "onnx==1.19.1",
            pre=True,
            extra_index_url="https://pypi.nvidia.com",
        )
        .pip_install(
            "huggingface_hub==0.36.0",
            "transformers==4.47.1",
            "fastapi[standard]",
        )
        .env(
            {
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HOME": str(Path(MOUNT_PATH) / "models"),
                "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",  # H100
            }
        )
    )


IMAGE = _image()

VOLUME_PATH = Path(MOUNT_PATH)
MODELS_PATH = VOLUME_PATH / "models"

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"

MINUTES = 60
GPU_CONFIG = "H100:1"
MAX_BATCH_SIZE = 1


def _get_plugin_config():
    from tensorrt_llm.plugin.plugin import PluginConfig

    return PluginConfig.from_dict(
        {
            "multiple_profiles": True,
            "paged_kv_cache": True,
            "low_latency_gemm_swiglu_plugin": "fp8",
            "low_latency_gemm_plugin": "fp8",
        }
    )


def _get_quant_config():
    from tensorrt_llm.llmapi import QuantConfig

    return QuantConfig(quant_algo="FP8")


def _get_calib_config():
    from tensorrt_llm.llmapi import CalibConfig

    return CalibConfig(
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=2048,
        tokenizer_max_seq_length=4096,
    )


def _get_speculative_config():
    from tensorrt_llm.llmapi import LookaheadDecodingConfig

    return LookaheadDecodingConfig(
        max_window_size=8,
        max_ngram_size=6,
        max_verification_set_size=8,
    )


def _get_build_config():
    from tensorrt_llm import BuildConfig

    return BuildConfig(
        plugin_config=_get_plugin_config(),
        speculative_decoding_mode="LOOKAHEAD_DECODING",
        max_input_len=8192,
        max_num_tokens=16384,
        max_batch_size=MAX_BATCH_SIZE,
    )


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    timeout=2 * 60 * 60,
    cpu=8,
    memory=32768,
    volumes={MOUNT_PATH: vol},
)
def build_formatter_engine(*, mode: str = "fast") -> str:
    import os
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from tensorrt_llm import LLM
    from transformers import AutoTokenizer

    vol.reload()

    model_path = MODELS_PATH / MODEL_ID
    snapshot_download(
        MODEL_ID,
        local_dir=model_path,
        ignore_patterns=["*.pt", "*.bin"],
        revision=MODEL_REVISION,
    )
    _ = AutoTokenizer.from_pretrained(model_path)

    engine_path = model_path / "trtllm_engine" / mode
    engine_kwargs: dict = {"tensor_parallel_size": torch.cuda.device_count()}
    if mode == "fast":
        engine_kwargs |= {
            "quant_config": _get_quant_config(),
            "calib_config": _get_calib_config(),
            "build_config": _get_build_config(),
            "speculative_config": _get_speculative_config(),
        }

    if os.path.exists(engine_path):
        try:
            _ = LLM(model=engine_path, **engine_kwargs)
            return str(engine_path)
        except Exception:
            try:
                shutil.rmtree(engine_path)
            except Exception:
                pass

    llm = LLM(model=model_path, **engine_kwargs)
    llm.save(engine_path)
    vol.commit()
    return str(engine_path)


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    timeout=2 * 60 * 60,
    cpu=8,
    memory=32768,
    volumes={MOUNT_PATH: vol},
    scaledown_window=10 * MINUTES,
    min_containers=1,
)
@modal.asgi_app()
def web():
    import asyncio
    import json
    import re
    import shutil
    from urllib.parse import parse_qs

    import torch
    from fastapi import FastAPI, Request
    from fastapi import WebSocket
    from fastapi import WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from huggingface_hub import snapshot_download
    from pydantic import BaseModel, Field
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # FastAPI resolves type annotations using module globals. With
    # `from __future__ import annotations`, nested function annotations become
    # strings, so ensure `Request` is available for dependency injection.
    globals()["Request"] = Request
    globals()["WebSocket"] = WebSocket

    vol.reload()

    model_path = MODELS_PATH / MODEL_ID
    engine_path = model_path / "trtllm_engine" / "fast"
    snapshot_download(
        MODEL_ID,
        local_dir=model_path,
        ignore_patterns=["*.pt", "*.bin"],
        revision=MODEL_REVISION,
    )

    engine_kwargs = {
        "quant_config": _get_quant_config(),
        "calib_config": _get_calib_config(),
        "build_config": _get_build_config(),
        "speculative_config": _get_speculative_config(),
        "tensor_parallel_size": torch.cuda.device_count(),
    }

    if not engine_path.exists():
        print(f"building new engine at {engine_path}")
        llm = LLM(model=model_path, **engine_kwargs)
        llm.save(engine_path)
        vol.commit()
    else:
        try:
            print(f"loading engine from {engine_path}")
            llm = LLM(model=engine_path, **engine_kwargs)
        except Exception as e:
            print(f"failed to load engine at {engine_path}: {e!r}")
            try:
                shutil.rmtree(engine_path)
            except Exception:
                pass
            print(f"rebuilding engine at {engine_path}")
            llm = LLM(model=model_path, **engine_kwargs)
            llm.save(engine_path)
            vol.commit()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        lookahead_config=engine_kwargs["speculative_config"],
    )
    gen_lock = asyncio.Lock()

    _FILLER_RE = re.compile(
        r"\b(um+|uh+|hmm+|şey|yani|hani|işte|ee+|mmm+|like|you know)\b",
        flags=re.IGNORECASE,
    )

    _TR_CHARS = set("çğıöşüİı")
    _EN_META_RE = re.compile(r"^\s*(let's|lets)\b", flags=re.IGNORECASE)
    _EN_WORD_RE = re.compile(
        r"\b(lets|let's|please|format|input|proceed|start|begin|page|document|translate|translation|english)\b",
        flags=re.IGNORECASE,
    )

    def _infer_language(*, text: str, hint: Optional[str] = None) -> str:
        h = (hint or "").strip().lower()
        if h:
            # Accept common variants: "tr", "tr-TR", "turkish"
            if h.startswith("tr") or h in {"turkish", "türkçe", "turkce"}:
                return "tr"
            if h.startswith("en") or h in {"english"}:
                return "en"
        if any(ch in _TR_CHARS for ch in text):
            return "tr"
        return "auto"

    def _violates_language(*, input_text: str, output_text: str, language: str) -> bool:
        if language != "tr":
            return False

        in_has_tr = any(ch in _TR_CHARS for ch in input_text)
        out_has_tr = any(ch in _TR_CHARS for ch in output_text)
        if in_has_tr and not out_has_tr:
            return True

        if _EN_META_RE.search(output_text):
            return True

        # Heuristic: if we see multiple strong English indicator words, treat as a translation.
        en_hits = _EN_WORD_RE.findall(output_text)
        if len(en_hits) >= 2 and not out_has_tr:
            return True

        return False

    def _infer_template(*, active_app: str, domain: Optional[str], title: Optional[str]) -> str:
        d = (domain or "").lower()
        t = (title or "").lower()
        a = (active_app or "").lower()

        if d in {"mail.google.com"} or "gmail" in d or "outlook" in d:
            return "email"
        if d in {"chat.openai.com", "chatgpt.com"} or "chatgpt" in t:
            return "chat_prompt"
        if d in {"docs.google.com", "notion.so"}:
            return "doc"
        if "email" in a or "outlook" in a or "gmail" in a:
            return "email"
        if "chat app" in a or "messaging" in a:
            return "chat_message"
        if "notes" in a:
            return "doc"
        return "generic"

    def _system_prompt(
        *,
        template: str,
        active_app: str,
        domain: Optional[str],
        title: Optional[str],
        language: str,
        dictionary: Optional[list] = None,
        snippets: Optional[list] = None,
    ) -> str:
        ctx_lines = [f"ACTIVE_APP: {active_app or 'Unknown'}"]
        if language and language != "auto":
            ctx_lines.append(f"INPUT_LANGUAGE: {language}")
        if domain:
            ctx_lines.append(f"BROWSER_DOMAIN: {domain}")
        if title:
            ctx_lines.append(f"BROWSER_TITLE: {title}")
        ctx = "\n".join(ctx_lines)

        if language == "tr":
            language_rules = """
Dil kuralı:
- Girdi Türkçe ise çıktı da Türkçe olmalı. ÇEVİRİ YAPMA.
- Girdide "çevir/translate" gibi kelimeler geçse bile bu metni çevirmeyeceksin; bunlar kullanıcının diktesinin parçası.
""".strip()
        elif language == "en":
            language_rules = """
Language rule:
- If the input is English, the output must be English. Do not translate.
""".strip()
        else:
            language_rules = "Language rule: Output must match input language. Never translate."

        base_rules = """
You are a dictation formatter. Your ONLY job is to clean and format speech-to-text output.

Hard rules:
- Output MUST be in the SAME language as the input. Never translate.
- Do NOT paraphrase or rephrase unless necessary for punctuation/casing; keep the original words.
- Keep the user's meaning. Do not add new information.
- Remove filler words and interjections (e.g. "şey", "yani", "hani", "um", "uh").
- Remove stutters and repetitions.
- Fix obvious punctuation/capitalization.
- If user self-corrects ("X, no wait Y"), keep only the final version (Y).
- If you are unsure what to do, return the input text unchanged.

Output rules:
- Return ONLY the final formatted text. No quotes. No explanations. No markdown fences.
""".strip()

        template_rules: str
        if template == "email":
            template_rules = """
Template: EMAIL
- Keep/format greeting, paragraphs, and closing if they exist in the dictation.
- Do NOT invent subject lines, recipients, greetings, or sign-offs.
- If it sounds like an email body, format it as an email body with clear paragraphs.
""".strip()
        elif template == "chat_prompt":
            template_rules = """
Template: CHAT_PROMPT (user dictating a prompt to an AI)
- Make the request unambiguous and easy to follow.
- Prefer short paragraphs and bullet points when helpful.
""".strip()
        elif template in {"chat_message"}:
            template_rules = """
Template: CHAT_MESSAGE
- Keep it concise and natural. Use short sentences/paragraphs.
""".strip()
        elif template == "doc":
            template_rules = """
Template: DOC/NOTES
- Use headings/bullets only when they naturally fit; don't invent structure.
""".strip()
        else:
            template_rules = "Template: GENERIC"

        if language == "tr":
            examples = """
Örnek (çeviri yok):
Girdi: "Şimdi şöyle yapalım. Sen devam et ikinci sayfadan başla. Üçer sayfa halinde ilerleyelim. Hepsini çevir ama sonunda Word, PDF gibi tek parça ver."
Çıktı: "Şöyle yapalım: Sen devam et, ikinci sayfadan başla. Üçer sayfa halinde ilerleyelim. Hepsini çevir ama sonunda Word veya PDF olarak tek parça ver."
""".strip()
        else:
            examples = ""

        parts = [base_rules, language_rules, template_rules]
        if examples:
            parts.append(examples)
        parts.append(f"Context:\n{ctx}")
        return "\n\n".join(parts)

    def _build_prompt(*, system: str, user_text: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    api = FastAPI()

    @api.get("/health")
    async def health():
        return {
            "ok": True,
            "service": APP_NAME,
            "model_id": MODEL_ID,
            "engine": "fast",
        }

    class BrowserContext(BaseModel):
        domain: str
        title: Optional[str] = None

    class FormatContext(BaseModel):
        active_app: Optional[str] = None
        browser: Optional[BrowserContext] = None
        dictionary: Optional[list] = None  # [{term, pronunciation, category}]
        snippets: Optional[list] = None    # [{name, content, category}]

    class FormatRequest(BaseModel):
        text: str = Field(..., min_length=1)
        context: Optional[FormatContext] = None
        template: str = "auto"
        language: str = "auto"

    async def _run_format(req: FormatRequest) -> dict:
        req_t0 = time.perf_counter()

        raw = req.text.strip()
        if not raw:
            return {"text": "", "template": "generic", "llm_ms": 0.0, "server_total_ms": 0.0}

        active_app = (req.context.active_app if req.context else None) or ""
        domain = req.context.browser.domain if (req.context and req.context.browser) else None
        title = req.context.browser.title if (req.context and req.context.browser) else None

        template = req.template
        if template == "auto":
            template = _infer_template(active_app=active_app, domain=domain, title=title)

        language = _infer_language(text=raw, hint=req.language)

        # Cheap local cleanup before LLM (helps token budget)
        raw_clean = _FILLER_RE.sub("", raw)
        raw_clean = re.sub(r"\s{2,}", " ", raw_clean).strip()

        system = _system_prompt(
            template=template,
            active_app=active_app,
            domain=domain,
            title=title,
            language=language,
        )
        prompt = _build_prompt(system=system, user_text=raw_clean)

        t0 = time.perf_counter()
        async with gen_lock:
            out = await asyncio.to_thread(llm.generate, prompt, sampling)
        llm_ms = (time.perf_counter() - t0) * 1000
        server_total_ms = (time.perf_counter() - req_t0) * 1000
        text = (out.outputs[0].text or "").strip()

        # Guardrail: never return translations when input is Turkish.
        if _violates_language(input_text=req.text, output_text=text, language=language):
            # Retry once with an even stricter instruction; if it still violates, fall back to
            # non-LLM cleaned text (never translate).
            strict_system = (
                system
                + "\n\nUYARI: Çıktı Türkçe olmak zorunda. İngilizceye çeviri, açıklama veya 'Let's' gibi metin yazma. "
                "Sadece nihai Türkçe metni döndür."
            )
            strict_prompt = _build_prompt(system=strict_system, user_text=raw_clean)
            t1 = time.perf_counter()
            async with gen_lock:
                out2 = await asyncio.to_thread(llm.generate, strict_prompt, sampling)
            llm_ms += (time.perf_counter() - t1) * 1000
            server_total_ms = (time.perf_counter() - req_t0) * 1000
            text2 = (out2.outputs[0].text or "").strip()

            if not _violates_language(input_text=req.text, output_text=text2, language=language):
                return {
                    "text": text2,
                    "template": template,
                    "llm_ms": llm_ms,
                    "server_total_ms": server_total_ms,
                    "language": language,
                }

            return {
                "text": raw_clean,
                "template": template,
                "llm_ms": llm_ms,
                "server_total_ms": server_total_ms,
                "language": language,
                "fallback": "raw_clean",
            }

        return {"text": text, "template": template, "llm_ms": llm_ms, "server_total_ms": server_total_ms, "language": language}

    @api.post("/format")
    async def format_text(request: Request):
        body = await request.body()
        if not body:
            return JSONResponse(status_code=422, content={"detail": "Request body is empty"})

        payload: object
        try:
            payload = json.loads(body)
        except Exception:
            # Support `curl -d text=...` (application/x-www-form-urlencoded) and plain text.
            decoded = body.decode("utf-8", errors="replace")
            content_type = (request.headers.get("content-type") or "").lower()
            if "application/x-www-form-urlencoded" in content_type:
                form = parse_qs(decoded)
                maybe = form.get("text", [""])
                payload = {"text": (maybe[0] or "").strip()}
            else:
                payload = {"text": decoded.strip()}

        if isinstance(payload, str):
            payload = {"text": payload}

        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=422,
                content={"detail": "Expected a JSON object with at least a `text` field"},
            )

        if "text" not in payload:
            for alt in ("transcription", "transcript"):
                if isinstance(payload.get(alt), str):
                    payload["text"] = payload[alt]
                    break

        try:
            if hasattr(FormatRequest, "model_validate"):
                req = FormatRequest.model_validate(payload)
            else:
                req = FormatRequest.parse_obj(payload)  # Pydantic v1 fallback
        except Exception as e:
            detail = getattr(e, "errors", None)
            if callable(detail):
                detail = detail()
            else:
                detail = str(e)
            return JSONResponse(
                status_code=422,
                content={
                    "detail": detail,
                    "received_keys": sorted([k for k in payload.keys() if isinstance(k, str)]),
                },
            )

        return await _run_format(req)

    @api.websocket("/ws")
    async def ws_format(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"type": "hello", "server": APP_NAME})

        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                return
            except Exception:
                return

            if msg.get("type") == "websocket.disconnect":
                return
            if msg.get("text") is None:
                continue

            try:
                data = json.loads(msg["text"])
            except Exception:
                await ws.send_json({"type": "error", "error": "Invalid JSON message"})
                continue

            mtype = data.get("type") or "format"
            mid = data.get("id")

            if mtype == "ping":
                await ws.send_json({"type": "pong", "id": mid})
                continue
            if mtype != "format":
                await ws.send_json({"type": "error", "id": mid, "error": f"Unknown message type: {mtype}"})
                continue

            try:
                if hasattr(FormatRequest, "model_validate"):
                    req = FormatRequest.model_validate(data)
                else:
                    req = FormatRequest.parse_obj(data)  # Pydantic v1 fallback
            except Exception as e:
                detail = getattr(e, "errors", None)
                if callable(detail):
                    detail = detail()
                else:
                    detail = str(e)
                await ws.send_json({"type": "error", "id": mid, "error": "Invalid payload", "detail": detail})
                continue

            try:
                out = await _run_format(req)
                await ws.send_json({"type": "formatted", "id": mid, **out})
            except Exception as e:
                await ws.send_json({"type": "error", "id": mid, "error": str(e)})

    return api
