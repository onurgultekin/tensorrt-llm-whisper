from pathlib import Path

import modal

from trtllm_whisper.build import BuildConfig, build_engines, convert_checkpoint

APP_NAME = "tensorrt-llm-whisper"
VOLUME_NAME = "trtllm-whisper"
MOUNT_PATH = "/vol"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _image():
    # CUDA base: keep <= host CUDA version (Modal guide recommends nvidia/cuda:*devel*)
    # Python: choose a version compatible with NVIDIA wheels (TRT-LLM commonly supports 3.10/3.11).
    return (
        # tensorrt_llm wheels currently link against CUDA 13 runtime libs (e.g. libcudart.so.13).
        modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu22.04", add_python="3.10")
        .apt_install(
            "git",
            "ffmpeg",
            "libsndfile1",
            # TensorRT-LLM python wheels `ctypes.LoadLibrary("libpython3.10.so")` ile bu shared lib'i arıyor.
            # Ubuntu'da `libpython3.10.so` symlink'i genelde `libpython3.10-dev` ile geliyor.
            "libpython3.10-dev",
            # TensorRT-LLM bindings import'u MPI runtime kütüphanesini (libmpi.so.40) bekliyor.
            "libopenmpi3",
            # MPI init hatalarını (özellikle root altında) azaltmak için OpenMPI yardımcı dosyaları/ikili dosyaları.
            "openmpi-bin",
            "openmpi-common",
        )
        .env(
            {
                # Modal container'ları root olarak çalışır; OpenMPI varsayılan olarak root'u reddedebilir.
                "OMPI_ALLOW_RUN_AS_ROOT": "1",
                "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            }
        )
        .pip_install(
            "numpy",
            "safetensors",
            "soundfile",
            "tiktoken",
            "fastapi[standard]",
            # Torch CUDA wheels
            "torch==2.5.1+cu124",
            extra_index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            # TensorRT-LLM wheels are hosted on NVIDIA's index.
            "tensorrt_llm",
            extra_index_url="https://pypi.nvidia.com",
        )
        # Bundle local code so the container can import `trtllm_whisper`.
        .add_local_python_source("trtllm_whisper")
    )


IMAGE = _image()

@app.function(image=IMAGE, timeout=10 * 60, cpu=2)
def debug_env() -> None:
    import subprocess
    import sys

    print("python:", sys.version)
    subprocess.run(["python3", "-m", "pip", "show", "tensorrt_llm"], check=False)
    subprocess.run(["bash", "-lc", "ldconfig -p | grep -E 'libcudart|libmpi' || true"], check=False)
    subprocess.run(["bash", "-lc", "ls -la /usr/local/cuda/lib64/libcudart.so* || true"], check=False)
    subprocess.run(["bash", "-lc", "which trtllm-build || true"], check=False)

    try:
        import importlib.metadata as md

        print("tensorrt_llm dist version:", md.version("tensorrt_llm"))
    except Exception as e:
        print("could not read tensorrt_llm dist version:", repr(e))

    try:
        import tensorrt_llm

        print("tensorrt_llm import OK:", tensorrt_llm.__version__)
    except Exception as e:
        print("tensorrt_llm import FAILED:", repr(e))


@app.function(image=IMAGE, timeout=10 * 60, cpu=2, gpu="A10")
def debug_env_gpu() -> None:
    import subprocess

    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(["bash", "-lc", "ldconfig -p | grep -E 'libcudart|libcuda|libmpi' || true"], check=False)
    subprocess.run(["bash", "-lc", "which trtllm-build || true"], check=False)

    try:
        import tensorrt_llm

        print("tensorrt_llm import OK:", tensorrt_llm.__version__)
    except Exception as e:
        print("tensorrt_llm import FAILED:", repr(e))


@app.function(
    image=IMAGE,
    gpu="A10",
    timeout=60 * 60 * 2,
    cpu=8,
    memory=32768,
    volumes={MOUNT_PATH: vol},
)
def build_whisper_engines(model_name: str = "large-v3") -> str:
    vol.reload()
    root = Path(MOUNT_PATH)
    assets_dir = root / "assets"
    checkpoints_dir = root / "checkpoints"
    engines_dir = root / "engines"

    cfg = BuildConfig(model_name=model_name)
    ckpt_dir = convert_checkpoint(assets_dir=assets_dir, checkpoints_dir=checkpoints_dir, cfg=cfg)
    engine_dir = build_engines(checkpoint_dir=ckpt_dir, engines_dir=engines_dir, cfg=cfg)
    vol.commit()
    return str(engine_dir)


@app.cls(
    image=IMAGE,
    gpu="A10",
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    volumes={MOUNT_PATH: vol},
    scaledown_window=10 * 60,
)
class WhisperService:
    model_name: str = modal.parameter(default="large-v3")

    @modal.enter()
    def _load(self):
        from trtllm_whisper.inference import WhisperTRTLLMRunner

        vol.reload()
        root = Path(MOUNT_PATH)
        cfg = BuildConfig(model_name=self.model_name)
        engine_dir = root / "engines" / cfg.build_id()
        assets_dir = root / "assets"
        if not (engine_dir / "encoder" / "rank0.engine").exists():
            raise RuntimeError(
                f"Engine not found at {engine_dir}. First run: "
                f"modal run modal_app.py::build_whisper_engines --model-name {self.model_name}"
            )

        self.runner = WhisperTRTLLMRunner(
            engine_dir=engine_dir,
            assets_dir=assets_dir,
            max_batch_size=cfg.max_batch_size,
            max_beam_width=cfg.max_beam_width,
        )

    @modal.method()
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        suffix: str = ".wav",
        text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        num_beams: int = 1,
        max_new_tokens: int = 96,
        return_timings: bool = False,
    ) -> str:
        from trtllm_whisper.inference import InferenceConfig

        cfg = InferenceConfig(text_prefix=text_prefix, num_beams=num_beams, max_new_tokens=max_new_tokens)
        if return_timings:
            return self.runner.transcribe_bytes_with_timings(audio_bytes, suffix=suffix, cfg=cfg)
        return self.runner.transcribe_bytes(audio_bytes, suffix=suffix, cfg=cfg)


@app.function(
    image=IMAGE,
    gpu="A10",
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    volumes={MOUNT_PATH: vol},
    scaledown_window=10 * 60,
)
@modal.asgi_app()
def web():
    import asyncio
    import json
    import time
    import uuid
    from typing import Optional

    from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from trtllm_whisper.inference import InferenceConfig, WhisperTRTLLMRunner

    vol.reload()
    root = Path(MOUNT_PATH)
    cfg = BuildConfig(model_name="large-v3")
    engine_dir = root / "engines" / cfg.build_id()
    assets_dir = root / "assets"
    if not (engine_dir / "encoder" / "rank0.engine").exists():
        raise RuntimeError(
            f"Engine not found at {engine_dir}. First run: "
            "modal run modal_app.py::build_whisper_engines --model-name large-v3"
        )

    runner = WhisperTRTLLMRunner(
        engine_dir=engine_dir,
        assets_dir=assets_dir,
        max_batch_size=cfg.max_batch_size,
        max_beam_width=cfg.max_beam_width,
    )

    inference_lock: Optional[asyncio.Lock] = None

    def _build_text_prefix(*, language: str, task: str, timestamps: bool) -> str:
        text_prefix = "<|startoftranscript|>"
        if language:
            text_prefix += f"<|{language}|>"
        text_prefix += f"<|{task}|>"
        if not timestamps:
            text_prefix += "<|notimestamps|>"
        return text_prefix

    async def _get_inference_lock() -> asyncio.Lock:
        nonlocal inference_lock
        if inference_lock is None:
            inference_lock = asyncio.Lock()
        return inference_lock

    api = FastAPI()

    @api.websocket("/ws")
    async def ws_transcribe(ws: WebSocket):
        await ws.accept()

        active = False
        buffer = bytearray()
        segments: list[str] = []

        sr = 16000
        fmt = "pcm_s16le"
        language = "en"
        task = "transcribe"
        timestamps = False
        num_beams = 1
        max_new_tokens = 96

        # Push-to-talk default: only emit results on explicit "end".
        # (Set >0 to enable auto-segmentation when audio pauses.)
        segment_silence_ms = 0
        min_segment_ms = 200
        last_voice_ts: Optional[float] = None
        flush_lock = asyncio.Lock()
        session_epoch = 0

        async def flush_segment(reason: str) -> Optional[dict]:
            nonlocal buffer, segments, last_voice_ts, session_epoch
            async with flush_lock:
                if not buffer:
                    return None
                if fmt != "pcm_s16le":
                    await ws.send_json({"type": "error", "error": f"Unsupported fmt: {fmt}"})
                    buffer = bytearray()
                    return None

                audio_sec = (len(buffer) / 2) / float(sr)
                if audio_sec * 1000 < min_segment_ms:
                    buffer = bytearray()
                    return None

                epoch = session_epoch
                pcm = bytes(buffer)
                buffer = bytearray()
                last_voice_ts = None

                cfg_local = InferenceConfig(
                    text_prefix=_build_text_prefix(language=language, task=task, timestamps=timestamps),
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                )

                lock = await _get_inference_lock()
                t0 = time.perf_counter()
                async with lock:
                    result = await asyncio.to_thread(
                        runner.transcribe_pcm16le_with_timings, pcm, sr=sr, cfg=cfg_local
                    )

                if epoch != session_epoch:
                    return None

                server_total_ms = (time.perf_counter() - t0) * 1000
                text = (result.get("text") or "").strip()
                if text:
                    segments.append(text)

                full_text = " ".join(s for s in segments if s).strip()
                timings_ms = dict(result.get("timings_ms") or {})
                timings_ms["server_total"] = server_total_ms
                if "total" in timings_ms:
                    timings_ms["server_overhead"] = max(0.0, server_total_ms - float(timings_ms["total"]))

                return {
                    "reason": reason,
                    "text": text,
                    "full_text": full_text,
                    "audio_sec": result.get("audio_sec"),
                    "rtf_cuda": result.get("rtf_cuda"),
                    "timings_ms": timings_ms,
                }

        async def segment_watcher() -> None:
            while True:
                await asyncio.sleep(0.05)
                if not active:
                    continue
                if segment_silence_ms <= 0:
                    continue
                if not buffer:
                    continue
                if last_voice_ts is None:
                    continue
                if (time.monotonic() - last_voice_ts) * 1000 >= segment_silence_ms:
                    payload = await flush_segment("silence")
                    if payload:
                        await ws.send_json({"type": "segment", **payload})

        watcher_task = asyncio.create_task(segment_watcher())

        try:
            await ws.send_json(
                {
                    "type": "hello",
                    "server": "tensorrt-llm-whisper",
                    "session_id": uuid.uuid4().hex,
                    "supported_formats": ["pcm_s16le"],
                }
            )

            while True:
                msg = await ws.receive()
                if msg.get("text") is not None:
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        await ws.send_json({"type": "error", "error": "Invalid JSON message"})
                        continue

                    mtype = data.get("type")
                    if mtype == "start":
                        session_epoch += 1
                        active = True
                        buffer = bytearray()
                        segments = []
                        last_voice_ts = None

                        sr = int(data.get("sr", sr))
                        fmt = str(data.get("fmt", fmt))
                        language = str(data.get("language", language))
                        task = str(data.get("task", task))
                        timestamps = bool(data.get("timestamps", timestamps))
                        num_beams = int(data.get("num_beams", num_beams))
                        max_new_tokens = int(data.get("max_new_tokens", max_new_tokens))

                        segment_silence_ms = int(data.get("segment_silence_ms", segment_silence_ms))
                        min_segment_ms = int(data.get("min_segment_ms", min_segment_ms))

                        if sr != 16000:
                            await ws.send_json(
                                {"type": "error", "error": f"Unsupported sample rate: {sr} (expected 16000)"}
                            )
                            active = False
                            continue
                        if fmt != "pcm_s16le":
                            await ws.send_json({"type": "error", "error": f"Unsupported fmt: {fmt}"})
                            active = False
                            continue

                        await ws.send_json(
                            {
                                "type": "started",
                                "sr": sr,
                                "fmt": fmt,
                                "segment_silence_ms": segment_silence_ms,
                                "min_segment_ms": min_segment_ms,
                                "segments": segments,
                            }
                        )
                    elif mtype == "end":
                        last = None
                        if active:
                            last = await flush_segment("end")
                        active = False
                        full_text = " ".join(s for s in segments if s).strip()
                        await ws.send_json({"type": "final", "text": full_text, "segments": segments, "last": last})
                        segments = []
                    elif mtype == "reset":
                        session_epoch += 1
                        active = False
                        buffer = bytearray()
                        segments = []
                        last_voice_ts = None
                        await ws.send_json({"type": "reset_ok"})
                    else:
                        await ws.send_json({"type": "error", "error": f"Unknown message type: {mtype}"})
                elif msg.get("bytes") is not None:
                    if not active:
                        continue

                    chunk = msg["bytes"]
                    if not isinstance(chunk, (bytes, bytearray)):
                        continue

                    max_bytes = 16000 * 30 * 2
                    if len(buffer) + len(chunk) > max_bytes:
                        _ = await flush_segment("max_len")

                    buffer.extend(chunk)
                    last_voice_ts = time.monotonic()
        except WebSocketDisconnect:
            return
        finally:
            watcher_task.cancel()

    @api.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        language: str = "en",
        task: str = "transcribe",
        timestamps: bool = False,
        num_beams: int = 1,
        max_new_tokens: int = 96,
        timings: bool = False,
    ):
        req_t0 = time.perf_counter()
        audio_bytes = await file.read()
        suffix = Path(file.filename or "").suffix or ".wav"
        text_prefix = "<|startoftranscript|>"
        if language:
            text_prefix += f"<|{language}|>"
        text_prefix += f"<|{task}|>"
        if not timestamps:
            text_prefix += "<|notimestamps|>"

        cfg = InferenceConfig(text_prefix=text_prefix, num_beams=num_beams, max_new_tokens=max_new_tokens)
        if timings:
            result = runner.transcribe_bytes_with_timings(audio_bytes, suffix=suffix, cfg=cfg)
            server_total_ms = (time.perf_counter() - req_t0) * 1000
            timings_ms = result.get("timings_ms", {})
            timings_ms["server_total"] = server_total_ms
            if "total" in timings_ms:
                timings_ms["server_overhead"] = max(0.0, server_total_ms - float(timings_ms["total"]))
            result["timings_ms"] = timings_ms
            return result
        text = runner.transcribe_bytes(audio_bytes, suffix=suffix, cfg=cfg)
        return {"text": text}

    return api


@app.local_entrypoint()
def main(
    audio_path: str,
    *,
    model_name: str = "large-v3",
    language: str = "en",
    task: str = "transcribe",
    timestamps: bool = False,
    num_beams: int = 1,
    max_new_tokens: int = 96,
    timings: bool = False,
):
    audio = Path(audio_path)
    audio_bytes = audio.read_bytes()
    suffix = audio.suffix or ".wav"
    text_prefix = "<|startoftranscript|>"
    if language:
        text_prefix += f"<|{language}|>"
    text_prefix += f"<|{task}|>"
    if not timestamps:
        text_prefix += "<|notimestamps|>"

    svc = WhisperService(model_name=model_name)
    text = svc.transcribe_bytes.remote(
        audio_bytes,
        suffix=suffix,
        text_prefix=text_prefix,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        return_timings=timings,
    )
    print(text)
