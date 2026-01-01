from pathlib import Path

import modal

from trtllm_whisper.build import BuildConfig, build_engines, convert_checkpoint

APP_NAME = "tensorrt-llm-whisper"
VOLUME_NAME = "trtllm-whisper"
MOUNT_PATH = "/vol"

LLM_VOLUME_NAME = "voiceos-text-formatter"
LLM_MOUNT_PATH = "/vol_llm"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
llm_vol = modal.Volume.from_name(LLM_VOLUME_NAME, create_if_missing=True)

GPU_CONFIG = "H100:1"

# How many concurrent WebSocket sessions a single container should serve before Modal
# spins up additional containers. This is critical to avoid "N users => N cold starts".
WS_MAX_INPUTS = 200


def _image():
    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .apt_install(
            "openmpi-bin",
            "libopenmpi-dev",
            "git",
            "git-lfs",
            "wget",
            "ffmpeg",
            "libsndfile1",
        )
        .env(
            {
                # Modal container'ları root olarak çalışır; OpenMPI varsayılan olarak root'u reddedebilir.
                "OMPI_ALLOW_RUN_AS_ROOT": "1",
                "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
                # Ensure PyTorch symbols are globally visible for TRT-LLM C++ bindings.
                "TORCH_USE_RTLD_GLOBAL": "YES",
                # Avoid JIT compiling for irrelevant archs.
                "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
                # Prevent Whisper from grabbing most of GPU RAM for paged KV cache.
                "VOICEOS_WHISPER_KV_CACHE_FRACTION": "0.2",
                "VOICEOS_WHISPER_CROSS_KV_CACHE_FRACTION": "0.5",
                # Prevent formatter LLM from grabbing most of GPU RAM for paged KV cache.
                "VOICEOS_FORMATTER_KV_CACHE_FRACTION": "0.12",
            }
        )
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
            "numpy",
            "safetensors",
            "soundfile",
            "tiktoken",
            "huggingface_hub==0.36.0",
            "transformers==4.47.1",
            "fastapi[standard]",
        )
        .run_commands(
            r"""python3 -c 'import site; from pathlib import Path; p=Path(site.getsitepackages()[0])/"sitecustomize.py"; p.write_text("import os\nimport sys\nimport ctypes\nos.environ.setdefault(\"TORCH_USE_RTLD_GLOBAL\",\"YES\")\nflags = sys.getdlopenflags()\nsys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)\ntry:\n    import torch\nexcept Exception:\n    pass\nfinally:\n    sys.setdlopenflags(flags)\n"); print("wrote", p)'"""
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
    subprocess.run(["python3", "-m", "pip", "show", "torch"], check=False)
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
        import torch

        print("torch import OK:", torch.__version__)
    except Exception as e:
        print("torch import FAILED:", repr(e))

    try:
        import tensorrt_llm

        print("tensorrt_llm import OK:", tensorrt_llm.__version__)
    except Exception as e:
        print("tensorrt_llm import FAILED:", repr(e))


@app.function(image=IMAGE, timeout=10 * 60, cpu=2, gpu=GPU_CONFIG)
def debug_env_gpu() -> None:
    import subprocess

    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(["bash", "-lc", "ldconfig -p | grep -E 'libcudart|libcuda|libmpi' || true"], check=False)
    subprocess.run(["bash", "-lc", "which trtllm-build || true"], check=False)

    try:
        import torch

        print("torch import OK:", torch.__version__)
        print("torch cuda device_count:", torch.cuda.device_count())
    except Exception as e:
        print("torch import FAILED:", repr(e))

    try:
        import tensorrt_llm

        print("tensorrt_llm import OK:", tensorrt_llm.__version__)
    except Exception as e:
        print("tensorrt_llm import FAILED:", repr(e))


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 2,
    cpu=8,
    memory=32768,
    volumes={MOUNT_PATH: vol},
)
def build_whisper_engines(model_name: str = "large-v3") -> str:
    import torch

    vol.reload()
    root = Path(MOUNT_PATH)
    assets_dir = root / "assets"
    checkpoints_dir = root / "checkpoints" / "trtllm018"
    major, minor = torch.cuda.get_device_capability()
    engines_dir = root / "engines" / f"sm{major}{minor}"

    cfg = BuildConfig(model_name=model_name)
    ckpt_dir = convert_checkpoint(assets_dir=assets_dir, checkpoints_dir=checkpoints_dir, cfg=cfg)
    engine_dir = build_engines(checkpoint_dir=ckpt_dir, engines_dir=engines_dir, cfg=cfg)
    vol.commit()
    return str(engine_dir)


LLM_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
LLM_MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"


def _env_float(name: str, default: str, *, min_value: float, max_value: float) -> float:
    import os

    raw = os.getenv(name, default)
    try:
        val = float(raw)
    except Exception as e:
        raise ValueError(f"{name} must be a float (got {raw!r})") from e
    if not (min_value <= val <= max_value):
        raise ValueError(f"{name} must be between {min_value} and {max_value} (got {val})")
    return val


def _formatter_kv_cache_fraction() -> float:
    # Keep small, otherwise TRT-LLM will try to allocate most of free GPU memory for KV cache.
    return _env_float("VOICEOS_FORMATTER_KV_CACHE_FRACTION", "0.12", min_value=0.01, max_value=0.95)


def _llm_models_path() -> Path:
    return Path(LLM_MOUNT_PATH) / "models"


def _llm_model_path() -> Path:
    return _llm_models_path() / LLM_MODEL_ID


def _llm_engine_path_legacy(*, mode: str = "fast") -> Path:
    return _llm_model_path() / "trtllm_engine" / mode


def _llm_engine_path(*, mode: str = "fast") -> Path:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return _llm_model_path() / "trtllm_engine" / f"sm{major}{minor}" / mode


def _llm_plugin_config():
    from tensorrt_llm.plugin.plugin import PluginConfig

    return PluginConfig.from_dict(
        {
            "multiple_profiles": True,
            "paged_kv_cache": True,
            "low_latency_gemm_swiglu_plugin": "fp8",
            "low_latency_gemm_plugin": "fp8",
        }
    )


def _llm_quant_config():
    from tensorrt_llm.llmapi import QuantConfig

    return QuantConfig(quant_algo="FP8")


def _llm_calib_config():
    from tensorrt_llm.llmapi import CalibConfig

    return CalibConfig(
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=2048,
        tokenizer_max_seq_length=4096,
    )


def _llm_speculative_config():
    from tensorrt_llm.llmapi import LookaheadDecodingConfig

    return LookaheadDecodingConfig(
        max_window_size=8,
        max_ngram_size=6,
        max_verification_set_size=8,
    )


def _llm_build_config(*, max_batch_size: int = 1):
    from tensorrt_llm import BuildConfig as TRTLLMBuildConfig

    return TRTLLMBuildConfig(
        plugin_config=_llm_plugin_config(),
        speculative_decoding_mode="LOOKAHEAD_DECODING",
        max_input_len=8192,
        max_num_tokens=16384,
        max_batch_size=max_batch_size,
    )


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    timeout=2 * 60 * 60,
    cpu=8,
    memory=32768,
    volumes={LLM_MOUNT_PATH: llm_vol},
)
def build_formatter_engine(*, mode: str = "fast") -> str:
    import os
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from tensorrt_llm import LLM
    from transformers import AutoTokenizer

    llm_vol.reload()

    model_path = _llm_model_path()
    snapshot_download(
        LLM_MODEL_ID,
        local_dir=model_path,
        ignore_patterns=["*.pt", "*.bin"],
        revision=LLM_MODEL_REVISION,
    )
    _ = AutoTokenizer.from_pretrained(model_path)

    engine_path = _llm_engine_path(mode=mode)
    engine_kwargs: dict = {"tensor_parallel_size": torch.cuda.device_count()}
    if mode == "fast":
        engine_kwargs |= {
            "quant_config": _llm_quant_config(),
            "calib_config": _llm_calib_config(),
            "build_config": _llm_build_config(max_batch_size=1),
            "speculative_config": _llm_speculative_config(),
        }

    kv_fraction = _formatter_kv_cache_fraction()

    if os.path.exists(engine_path):
        try:
            try:
                _ = LLM(model=engine_path, kv_cache_free_gpu_memory_fraction=kv_fraction, **engine_kwargs)
            except TypeError as e:
                # Fallback for TRT-LLM API changes: ignore kv cache fraction kwarg.
                if "kv_cache_free_gpu_memory_fraction" not in str(e):
                    raise
                _ = LLM(model=engine_path, **engine_kwargs)
            return str(engine_path)
        except Exception:
            try:
                shutil.rmtree(engine_path)
            except Exception:
                pass

    # KV cache fraction is mainly a runtime concern, but passing it keeps allocations sane if
    # TRT-LLM initializes executor components eagerly.
    try:
        llm = LLM(model=model_path, kv_cache_free_gpu_memory_fraction=kv_fraction, **engine_kwargs)
    except TypeError as e:
        if "kv_cache_free_gpu_memory_fraction" not in str(e):
            raise
        llm = LLM(model=model_path, **engine_kwargs)
    llm.save(engine_path)
    llm_vol.commit()
    return str(engine_path)


@app.cls(
    image=IMAGE,
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    volumes={MOUNT_PATH: vol},
    scaledown_window=10 * 60
)
class WhisperService:
    model_name: str = modal.parameter(default="large-v3")

    @modal.enter()
    def _load(self):
        import torch

        from trtllm_whisper.inference import WhisperTRTLLMRunner

        vol.reload()
        root = Path(MOUNT_PATH)
        cfg = BuildConfig(model_name=self.model_name)
        major, minor = torch.cuda.get_device_capability()
        engine_dir = root / "engines" / f"sm{major}{minor}" / cfg.build_id()
        assets_dir = root / "assets"
        if not (engine_dir / "encoder" / "rank0.engine").exists() or not (
            engine_dir / "decoder" / "rank0.engine"
        ).exists():
            raise RuntimeError(
                f"Engine not found at {engine_dir}. First run: "
                f"modal run modal_app.py::build_whisper_engines --model-name {self.model_name}"
            )

        self.runner = WhisperTRTLLMRunner(
            engine_dir=engine_dir,
            assets_dir=assets_dir,
            max_batch_size=cfg.max_batch_size,
            max_output_len=cfg.max_output_len,
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
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    cpu=8,  # Increased for better concurrency handling
    memory=32768,
    volumes={MOUNT_PATH: vol, LLM_MOUNT_PATH: llm_vol},
    scaledown_window=10 * 60,
    min_containers=1
)
@modal.concurrent(max_inputs=WS_MAX_INPUTS)
@modal.asgi_app()
def web():
    import asyncio
    import os
    import json
    import re
    import time
    import uuid
    import unicodedata
    from typing import Optional

    import torch
    from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from tensorrt_llm import LLM, SamplingParams
    from trtllm_whisper.inference import InferenceConfig, WhisperTRTLLMRunner
    from transformers import AutoTokenizer

    root = Path(MOUNT_PATH)
    cfg = BuildConfig(model_name="large-v3")
    engine_max_output_len = cfg.max_output_len
    major, minor = torch.cuda.get_device_capability()
    engine_dir = root / "engines" / f"sm{major}{minor}" / cfg.build_id()
    assets_dir = root / "assets"
    runner: Optional[WhisperTRTLLMRunner] = None
    runner_error: Optional[str] = None

    # No inference lock - TensorRT-LLM handles inflight batching internally
    # Multiple concurrent requests will be batched automatically

    llm: Optional[LLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    sampling: Optional[SamplingParams] = None
    gen_lock = asyncio.Lock()

    llm_init_error: Optional[str] = None

    def _build_text_prefix(*, language: str, task: str, timestamps: bool) -> str:
        text_prefix = "<|startoftranscript|>"
        if language:
            text_prefix += f"<|{language}|>"
        text_prefix += f"<|{task}|>"
        if not timestamps:
            text_prefix += "<|notimestamps|>"
        return text_prefix

    _WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

    def _normalize_word(word: str, *, language: str) -> str:
        w = unicodedata.normalize("NFKC", word)
        # Turkish dotted/dotless i case mapping for more reliable overlap detection.
        if language.lower().startswith("tr"):
            w = w.replace("I", "ı").replace("İ", "i").lower()
            return w
        return w.casefold()

    def _word_spans(text: str, *, language: str) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        for m in _WORD_RE.finditer(text):
            raw = m.group(0)
            spans.append((_normalize_word(raw, language=language), m.start(), m.end()))
        return spans

    def _words_match(a: str, b: str) -> bool:
        if a == b:
            return True
        # Handle suffixy languages (e.g. Turkish): "politikaları" vs "politikalarının".
        if len(a) >= 6 and len(b) >= 6:
            if a.startswith(b) or b.startswith(a):
                shorter = min(len(a), len(b))
                longer = max(len(a), len(b))
                if shorter / longer >= 0.75:
                    return True
        return False

    def _find_overlap_suffix_prefix(
        prev_text: str,
        new_text: str,
        *,
        language: str,
        min_words: int = 6,
        max_words: int = 80,
    ) -> tuple[int, int]:
        prev_spans = _word_spans(prev_text, language=language)
        new_spans = _word_spans(new_text, language=language)
        if not prev_spans or not new_spans:
            return 0, -1

        max_k = min(max_words, len(prev_spans), len(new_spans))
        for k in range(max_k, min_words - 1, -1):
            ok = True
            for i in range(k):
                if not _words_match(prev_spans[-k + i][0], new_spans[i][0]):
                    ok = False
                    break
            if ok:
                return k, prev_spans[-k][1]

        return 0, -1

    # Simple RMS-based voice activity detection (no external dependencies)
    def _has_voice_activity(pcm_bytes: bytes, sr: int = 16000, rms_threshold: float = 50.0, min_speech_ratio: float = 0.02) -> bool:
        """Check if audio has significant voice activity using RMS energy.
        
        Args:
            pcm_bytes: Raw PCM16LE audio bytes
            sr: Sample rate (unused, for API compatibility)
            rms_threshold: RMS threshold for speech detection (default 50, lowered for sensitivity)
            min_speech_ratio: Minimum ratio of speech frames (default 2%)
        """
        try:
            import numpy as np
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            
            if len(audio) == 0:
                return False
            
            # Calculate RMS in 30ms windows
            window_size = int(0.03 * 16000)  # 30ms at 16kHz
            if len(audio) < window_size:
                rms = np.sqrt(np.mean(audio ** 2))
                return rms > rms_threshold
            
            # Count windows with speech
            n_windows = len(audio) // window_size
            speech_windows = 0
            max_rms = 0.0
            
            for i in range(n_windows):
                window = audio[i * window_size:(i + 1) * window_size]
                rms = np.sqrt(np.mean(window ** 2))
                max_rms = max(max_rms, rms)
                if rms > rms_threshold:
                    speech_windows += 1
            
            speech_ratio = speech_windows / n_windows if n_windows > 0 else 0
            has_speech = speech_ratio >= min_speech_ratio
            
            print(f"🔊 VAD: max_rms={max_rms:.1f}, speech_ratio={speech_ratio:.2%}, has_speech={has_speech}")
            return has_speech
        except Exception as e:
            print(f"⚠️  VAD check failed: {e}")
            return True  # On error, assume voice is present

    api = FastAPI()

    def _init_models() -> None:
        nonlocal runner, runner_error, llm, tokenizer, sampling, llm_init_error

        vol.reload()
        llm_vol.reload()

        runner = None
        runner_error = None
        llm = None
        tokenizer = None
        sampling = None
        llm_init_error = None

        # Whisper runner
        if not (engine_dir / "encoder" / "rank0.engine").exists() or not (
            engine_dir / "decoder" / "rank0.engine"
        ).exists():
            runner_error = (
                f"Engine not found at {engine_dir}. First run: "
                "modal run modal_app.py::build_whisper_engines --model-name large-v3"
            )
            print(f"⚠️  {runner_error}")
        else:
            try:
                runner = WhisperTRTLLMRunner(
                    engine_dir=engine_dir,
                    assets_dir=assets_dir,
                    max_batch_size=cfg.max_batch_size,
                    max_output_len=cfg.max_output_len,
                    max_beam_width=cfg.max_beam_width,
                )

                print("🔥 Warming up Whisper engine...")
                import numpy as np

                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                dummy_pcm = (dummy_audio * 32768.0).astype(np.int16).tobytes()
                try:
                    runner.transcribe_pcm16le(dummy_pcm, sr=16000)
                    print("✅ Whisper warm-up complete")
                except Exception as e:
                    print(f"⚠️  Whisper warm-up failed (non-critical): {e}")
            except Exception as e:
                runner = None
                runner_error = repr(e)
                print(f"⚠️  Failed to init Whisper runner: {runner_error}")

        # Formatter LLM
        try:
            llm_engine_path = _llm_engine_path(mode="fast")
            legacy_llm_engine_path = _llm_engine_path_legacy(mode="fast")
            llm_model_path = _llm_model_path()

            if not llm_engine_path.exists():
                llm_init_error = (
                    f"LLM engine not found at {llm_engine_path}. First run: "
                    "modal run modal_app.py::build_formatter_engine --mode fast "
                    "(run from the tensorrt-llm-whisper directory, or use "
                    "modal run tensorrt-llm-whisper/modal_app.py::build_formatter_engine --mode fast)"
                )
                if legacy_llm_engine_path.exists():
                    llm_init_error += (
                        f" (legacy engine exists at {legacy_llm_engine_path}, but engines are GPU-specific)"
                    )
                print(f"⚠️  {llm_init_error}")
                return

            if not llm_model_path.exists():
                llm_init_error = f"LLM model dir not found at {llm_model_path}"
                print(f"⚠️  {llm_init_error}")
                return

            kv_fraction = _formatter_kv_cache_fraction()
            os.environ.setdefault("TLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION", str(kv_fraction))

            engine_kwargs = {
                "quant_config": _llm_quant_config(),
                "calib_config": _llm_calib_config(),
                "build_config": _llm_build_config(max_batch_size=1),
                "speculative_config": _llm_speculative_config(),
                "tensor_parallel_size": torch.cuda.device_count(),
            }

            # Help reduce fragmentation before loading the second large engine.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            print(f"🧠 Loading formatter LLM engine from {llm_engine_path} (kv_cache_fraction={kv_fraction}) ...")
            try:
                llm = LLM(model=llm_engine_path, kv_cache_free_gpu_memory_fraction=kv_fraction, **engine_kwargs)
            except TypeError as e:
                if "kv_cache_free_gpu_memory_fraction" not in str(e):
                    raise
                llm = LLM(model=llm_engine_path, **engine_kwargs)

            tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
            sampling = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=1024,
                lookahead_config=engine_kwargs["speculative_config"],
            )
            print("✅ Formatter LLM ready")

            # Warm-up: first generate is often much slower (allocator + kernel init).
            # Keep it tiny so startup cost stays low but the first user doesn't pay the latency spike.
            try:
                warm_messages = [
                    {"role": "system", "content": "You are a dictation editor."},
                    {"role": "user", "content": "Warmup."},
                ]
                warm_prompt = tokenizer.apply_chat_template(
                    warm_messages, tokenize=False, add_generation_prompt=True
                )
                warm_sampling = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=8,
                    lookahead_config=engine_kwargs["speculative_config"],
                )
                t_warm0 = time.perf_counter()
                _ = llm.generate(warm_prompt, warm_sampling)
                warm_ms = (time.perf_counter() - t_warm0) * 1000
                print(f"✅ Formatter warm-up complete ({warm_ms:.0f}ms)")
            except Exception as e:
                print(f"⚠️  Formatter warm-up failed (non-critical): {e}")
        except Exception as e:
            msg = str(e)
            lowered = msg.lower()
            if "out of memory" in lowered or "cuda runtime" in lowered:
                llm_init_error = (
                    "Formatter LLM failed to load due to GPU OOM. "
                    "Lower VOICEOS_FORMATTER_KV_CACHE_FRACTION and/or VOICEOS_WHISPER_KV_CACHE_FRACTION, then redeploy. "
                    f"(error: {msg})"
                )
            elif "Failed to deserialize cuda engine" in msg:
                llm_init_error = (
                    f"Failed to load LLM engine at {llm_engine_path}. "
                    "This can happen if the engine was built for a different GPU/TensorRT, or if GPU memory is exhausted. "
                    "Try rebuilding and/or lowering VOICEOS_FORMATTER_KV_CACHE_FRACTION. "
                    "Rebuild: modal run modal_app.py::build_formatter_engine --mode fast "
                    "(run from the tensorrt-llm-whisper directory, or use "
                    "modal run tensorrt-llm-whisper/modal_app.py::build_formatter_engine --mode fast)"
                )
            else:
                llm_init_error = repr(e)

            llm = None
            tokenizer = None
            sampling = None
            print(f"⚠️  Formatter LLM init failed: {llm_init_error}")

    @api.on_event("startup")
    async def _startup() -> None:
        nonlocal runner_error, llm_init_error
        try:
            _init_models()
        except Exception as e:
            if runner_error is None:
                runner_error = f"Startup init failed: {e!r}"
            if llm_init_error is None:
                llm_init_error = f"Startup init failed: {e!r}"

    FORMATTER_PROMPT_VERSION = "dictation_v2"
    _WORD_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
    _SPACE_RE = re.compile(r"[ \t]+")
    _MANY_NEWLINES_RE = re.compile(r"\n{3,}")
    _NUMBERED_LIST_ITEM_RE = re.compile(r"\b(\d{1,3})\s*[-–—.)]\s+", flags=re.UNICODE)

    def _normalize_ws(text: str) -> str:
        t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        t = _MANY_NEWLINES_RE.sub("\n\n", t)
        t = "\n".join(_SPACE_RE.sub(" ", line).strip() for line in t.split("\n"))
        return t.strip()

    # Common filler words across languages (for preprocessing/analysis)
    FILLER_WORDS = {
        # English
        "um", "uh", "er", "ah", "like", "you know", "i mean", "sort of", "kind of", "basically", "actually",
        # Turkish
        "yani", "işte", "şey", "hani", "falan", "filan", "böyle", "yani şey", "işte şey",
        # Spanish
        "eh", "este", "pues", "o sea", "bueno", "entonces",
        # French
        "euh", "ben", "alors", "donc", "quoi", "enfin",
        # German
        "äh", "ähm", "also", "halt", "sozusagen", "quasi",
        # Italian
        "ehm", "cioè", "allora", "dunque", "insomma",
        # Portuguese
        "né", "tipo", "então", "quer dizer",
        # Russian (transliterated)
        "nu", "vot", "tak", "znachit",
        # Japanese (transliterated)
        "ano", "eto", "ma",
        # Korean (transliterated)
        "geu", "eumm",
        # Arabic (transliterated)
        "yani", "yaani",
        # Chinese (transliterated)
        "nage", "jiushi", "ranhou",
    }

    def _trim_llm_preamble(text: str) -> str:
        """Remove common LLM preambles like 'Here is...', 'Here's...', etc."""
        t = text.strip()
        if not t:
            return t
        
        # Common preamble patterns (case-insensitive)
        preambles = [
            "here is the cleaned text:",
            "here's the cleaned text:",
            "here is the polished text:",
            "here's the polished text:",
            "here is the corrected text:",
            "here's the corrected text:",
            "here is the text:",
            "here's the text:",
            "here is:",
            "here's:",
            "cleaned text:",
            "polished text:",
            "corrected text:",
        ]
        
        t_lower = t.lower()
        for preamble in preambles:
            if t_lower.startswith(preamble):
                # Remove the preamble and any following whitespace/newlines
                t = t[len(preamble):].lstrip()
                break
        
        return t

    def _word_tokens(text: str) -> list[str]:
        """Extract normalized word tokens, handling Turkish and other agglutinative languages."""
        tokens = []
        for m in _WORD_TOKEN_RE.finditer(text):
            word = unicodedata.normalize("NFKC", m.group(0))
            # Remove apostrophes and hyphens for comparison (slide'teki → slideteki)
            normalized = word.replace("'", "").replace("'", "").replace("-", "").casefold()
            tokens.append(normalized)
        return tokens

    def _is_subsequence(source: list[str], target: list[str]) -> bool:
        if not target:
            return True
        i = 0
        for tok in source:
            if tok == target[i]:
                i += 1
                if i >= len(target):
                    return True
        return False

    def _is_morphological_variant(word1: str, word2: str) -> bool:
        """Check if two words are morphological variants (e.g., Turkish suffixes)."""
        if word1 == word2:
            return True
        # If one is a substring of the other and they share significant prefix
        if len(word1) >= 4 and len(word2) >= 4:
            shorter = min(word1, word2, key=len)
            longer = max(word1, word2, key=len)
            if longer.startswith(shorter) and len(shorter) / len(longer) >= 0.6:
                return True
        return False

    def _safety_check(*, input_text: str, output_text: str, allow_reordering: bool = False, dictionary_terms: Optional[list] = None) -> dict:
        in_tokens = _word_tokens(input_text)
        out_tokens = _word_tokens(output_text)
        in_set = set(in_tokens)
        
        # Build whitelist from dictionary terms (these are allowed even if not in input)
        dict_whitelist = set()
        if dictionary_terms:
            for entry in dictionary_terms:
                term = entry.get("term", "") if isinstance(entry, dict) else ""
                if term:
                    # Add all word tokens from the term to whitelist (lowercased for matching)
                    dict_whitelist.update(t.lower() for t in _word_tokens(term))
        
        # Check for truly new tokens (not morphological variants or dictionary terms)
        out_new = []
        for out_tok in out_tokens:
            if out_tok in in_set:
                continue
            # Check if it's in dictionary whitelist (case-insensitive)
            if out_tok.lower() in dict_whitelist:
                continue
            # Check if it's a morphological variant of any input token
            is_variant = any(_is_morphological_variant(out_tok, in_tok) for in_tok in in_tokens)
            if not is_variant:
                out_new.append(out_tok)
        
        # For subsequence check, filter out dictionary terms from output
        # (they can appear anywhere since they're replacements)
        out_tokens_for_subseq = [t for t in out_tokens if t.lower() not in dict_whitelist]
        
        # For formatting contexts (email, doc), allow reordering as long as no new words
        if allow_reordering:
            # Check if all output tokens exist in input (allowing reordering)
            all_tokens_present = all(
                out_tok.lower() in dict_whitelist or any(_is_morphological_variant(out_tok, in_tok) for in_tok in in_tokens)
                for out_tok in out_tokens
            )
            ok = bool(output_text.strip()) and (len(out_new) == 0) and all_tokens_present
            return {
                "ok": ok,
                "subsequence": False,  # Not checking subsequence when reordering allowed
                "in_token_count": len(in_tokens),
                "out_token_count": len(out_tokens),
                "out_new_token_count": len(out_new),
                "out_new_tokens_sample": out_new[:12],
                "allow_reordering": True,
            }
        
        # Standard strict check (for chat, generic, etc.)
        # Use filtered tokens for subsequence (excluding dictionary terms)
        subseq_ok = _is_subsequence(in_tokens, out_tokens_for_subseq)
        ok = bool(output_text.strip()) and (len(out_new) == 0) and subseq_ok
        return {
            "ok": ok,
            "subsequence": subseq_ok,
            "in_token_count": len(in_tokens),
            "out_token_count": len(out_tokens),
            "out_new_token_count": len(out_new),
            "out_new_tokens_sample": out_new[:12],
            "allow_reordering": False,
        }

    def _maybe_format_numbered_list(text: str) -> tuple[str, dict]:
        """
        Deterministic preprocessing for common dictation patterns like:
          "1- Apples 2- Bananas 3- Oranges"
        → a multi-line numbered list.

        This is language-agnostic (numbers), and does not add new words (only punctuation/newlines).
        """
        matches = list(_NUMBERED_LIST_ITEM_RE.finditer(text))
        if len(matches) < 2:
            return text, {"numbered_list": False}

        nums: list[int] = []
        for m in matches:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                return text, {"numbered_list": False}

        # Heuristic: require strictly increasing (typical "1, 2, 3...").
        if any(nums[i] >= nums[i + 1] for i in range(len(nums) - 1)):
            return text, {"numbered_list": False}

        prefix = text[: matches[0].start()].strip()
        items: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            item = text[start:end].strip()
            if not item:
                continue
            items.append((nums[i], item))

        if len(items) < 2:
            return text, {"numbered_list": False}

        out_lines: list[str] = []
        if prefix:
            out_lines.append(prefix)
        for n, item in items:
            out_lines.append(f"{n}. {item}")

        formatted = "\n".join(out_lines).strip()
        if not formatted:
            return text, {"numbered_list": False}

        return formatted, {"numbered_list": True, "items": len(items)}

    def _preprocess_llm_input(text: str) -> tuple[str, dict]:
        processed, list_info = _maybe_format_numbered_list(text)
        info: dict = {}
        if list_info.get("numbered_list"):
            info["numbered_list"] = list_info
        return processed, info

    def _trim_output_to_safe_candidate(*, input_text: str, output_text: str) -> Optional[str]:
        """
        Try to salvage a model output that added a preamble/epilogue by trimming whole lines.
        Only returns a candidate that passes _safety_check against input_text.
        """
        out = _normalize_ws(output_text)
        if not out:
            return None

        if _safety_check(input_text=input_text, output_text=out).get("ok"):
            return out

        in_set = set(_word_tokens(input_text))
        lines = out.split("\n")

        def line_has_new_tokens(line: str) -> bool:
            toks = _word_tokens(line)
            return any(t not in in_set for t in toks)

        # Trim leading junk (often "Sure, here's...").
        start = 0
        while start < len(lines):
            line = lines[start].strip()
            if not line:
                start += 1
                continue
            if line_has_new_tokens(line):
                start += 1
                continue
            break

        # Trim trailing junk (often a short explanation after the answer).
        end = len(lines)
        while end > start:
            line = lines[end - 1].strip()
            if not line:
                end -= 1
                continue
            if line_has_new_tokens(line):
                end -= 1
                continue
            break

        candidate = "\n".join(lines[start:end]).strip()
        if not candidate:
            return None

        if _safety_check(input_text=input_text, output_text=candidate).get("ok"):
            return candidate

        return None

    def _infer_language(*, text: str, hint: Optional[str] = None) -> str:
        h = (hint or "").strip()
        if not h:
            return "auto"
        h_low = h.lower()
        if h_low in {"auto", "unknown"}:
            return "auto"
        # Normalize "en-US" -> "en" (keep others as-is).
        return h.split("-", 1)[0].lower()

    def _infer_template(*, active_app: str, domain: Optional[str], title: Optional[str]) -> str:
        d = (domain or "").lower()
        t = (title or "").lower()
        a = (active_app or "").lower()

        # Code/IDE contexts
        if any(x in a for x in ["cursor", "windsurf", "vscode", "visual studio code", "code", "sublime", "atom", "vim", "neovim", "intellij", "pycharm", "webstorm"]):
            return "code"
        if any(x in d for x in ["github.com", "gitlab.com", "replit.com", "codesandbox.io", "stackblitz.com"]):
            return "code"
        if any(x in t for x in ["cursor", "windsurf", "vscode", "code editor", "ide"]):
            return "code"

        # Email contexts
        if d in {"mail.google.com"} or "gmail" in d or "outlook" in d or "mail.yahoo" in d:
            return "email"
        if "email" in a or "outlook" in a or "gmail" in a or "mail" in a:
            return "email"

        # Chat/messaging contexts
        if d in {"chat.openai.com", "chatgpt.com", "claude.ai", "web.whatsapp.com", "web.telegram.org"}:
            return "chat_prompt"
        if "chatgpt" in t or "claude" in t or "gemini" in t:
            return "chat_prompt"
        if any(x in a for x in ["whatsapp", "telegram", "slack", "discord", "teams", "messenger"]):
            return "chat_message"
        if "chat app" in a or "messaging" in a:
            return "chat_message"

        # Document contexts
        if d in {"docs.google.com", "notion.so", "coda.io", "airtable.com"}:
            return "doc"
        if any(x in a for x in ["word", "pages", "notes", "notion", "obsidian", "roam"]):
            return "doc"
        if "notes" in a or "document" in a:
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
        tone: Optional[dict] = None,
    ) -> str:
        ctx_lines = [f"ACTIVE_APP: {active_app or 'Unknown'}"]
        if language and language != "auto":
            ctx_lines.append(f"LANGUAGE_HINT: {language}")
        if domain:
            ctx_lines.append(f"BROWSER_DOMAIN: {domain}")
        if title:
            ctx_lines.append(f"BROWSER_TITLE: {title}")
        ctx = "\n".join(ctx_lines)
        
        # Build tone/style rules if provided
        tone_rules = ""
        if tone and tone.get("promptModifier"):
            tone_name = tone.get("name", "Custom")
            tone_modifier = tone.get("promptModifier", "")
            tone_rules = f"\n\nIMPORTANT - STYLE/TONE ({tone_name}):\nApply these style rules to the output:\n{tone_modifier}"
        
        # Build dictionary rules if provided
        dictionary_rules = ""
        if dictionary and len(dictionary) > 0:
            dict_items = []
            preserve_terms = []
            for entry in dictionary[:50]:  # Limit to 50 entries to avoid token overflow
                term = entry.get("term", "")
                pronunciation = entry.get("pronunciation", "")
                if term:
                    if pronunciation:
                        # Has pronunciation mapping
                        dict_items.append(f'  - "{pronunciation}" → "{term}"')
                    else:
                        # Just a term to preserve (no pronunciation)
                        preserve_terms.append(f'"{term}"')
            
            rules_parts = []
            if dict_items:
                rules_parts.append("Sound mappings (use exact spelling):\n" + "\n".join(dict_items))
            if preserve_terms:
                rules_parts.append("Preserve these terms exactly (do NOT change spelling): " + ", ".join(preserve_terms))
            
            if rules_parts:
                dictionary_rules = "\n\nCUSTOM VOCABULARY:\n" + "\n".join(rules_parts)

        template_rules = {
            "email": (
                "Email format. MUST structure as professional email:\n"
                "- If multiple action items/tasks mentioned → format as bullet list (• or numbered)\n"
                "- Organize into clear paragraphs (use blank lines)\n"
                "- Convert time references: noon→12 PM, morning→9 AM\n"
                "- Professional tone but keep natural\n"
                "- Do NOT invent greetings/signatures unless dictated"
            ),
            "chat_prompt": "Direct chat message. Conversational. Remove unnecessary politeness.",
            "chat_message": "Casual chat. Keep informal tone.",
            "doc": (
                "Document format. Structure clearly:\n"
                "- Organize into paragraphs\n"
                "- Format lists when appropriate\n"
                "- Professional structure"
            ),
            "code": "Preserve exact casing (camelCase, snake_case, etc.), dev tools (Supabase, Vercel, etc.), file paths, syntax.",
            "generic": "Clean natural text.",
        }.get(template, "Clean natural text.")

        # Ultra-focused dictation prompt - NO room for "helpful assistant" behavior
        base_rules = f"""You are a dictation text cleaner. Input = raw speech transcript. Output = cleaned text ONLY.

CRITICAL RULES:
1. Output the EXACT SAME LANGUAGE as input. NEVER translate.
2. Use ONLY words from the input (you may remove words, add punctuation/newlines/formatting).
3. For agglutinative languages (Turkish, Finnish, etc.): You may adjust suffixes for grammar, but keep root words.
4. NO explanations. NO preambles. NO "Here is..." or "Here's the...". NO markdown. NO quotes around output.
5. Output STARTS with the actual cleaned text, nothing before it.

ALLOWED EDITS:
• Remove fillers: um, uh, er, ah, like, you know, I mean, yani, işte, şey, hani, falan, eh, este, pues, euh, äh, ähm (and equivalents in all languages)
• Remove stutters: "I I I think" → "I think"
• Remove repetitions: "the the problem" → "the problem"
• Handle self-corrections (keep FINAL version only):
  - "meet at 2... actually 3" → "meet at 3"
  - "send to John... no wait Sarah" → "send to Sarah"
  - Recognize: actually, I mean, no wait, sorry, yok, aslında, değil, etc.
• Fix punctuation & capitalization (don't change words)
• Add apostrophes for clarity: "slideteki" → "slide'teki", "4teki" → "4'teki"
• Convert spoken punctuation: "comma"→, "period"→. "question mark"→? "virgül"→, "nokta"→. etc.
• Format numbered lists: "1 apples 2 bananas" → "1. Apples\n2. Bananas"
• Format action items as bullets when multiple tasks mentioned
• Normalize time expressions: "noon"→"12 PM", "midnight"→"12 AM" (use words from input)
• Structure into paragraphs when natural topic breaks occur

EXAMPLES:

Input: "Hi Nick. What I'd actually like to do here is conduct a bit more testing, if possible."
Output: "Hi Nick,

What I actually want to do here is conduct a bit more testing, if possible."

Input: "Hey um for tomorrow's meeting we need to finish the deck design has two slides left also check slide 4 numbers let's send Rachel the final copy before noon"
Output: "For tomorrow's meeting:

• Finish the deck — design has two slides left
• Check slide 4 numbers
• Send Rachel the final copy before 12 PM"

Input: "Selamlar yarınki toplantı için sunumu kontrol ederim dördüncü slidedeki rakamlar yanlış olabilir dizayn kısmında eksikler vardı onları da giderelim bir de Rachel onları bana tekrar geri göndersin"
Output: "Selamlar. Yarınki toplantı için:

• Sunumu kontrol edeceğim
• Dördüncü slide'deki rakamlar yanlış olabilir
• Dizayn kısmında eksikler vardı — onları giderelim
• Rachel onları bana tekrar göndersin"

Input: "Yarınki toplantı için şu üç maddeyi bitirmemiz gerekiyor Ahmet Mehmet'i arasın Mehmet Berfin'i arasın Berfin de beni arasın"
Output: "Yarınki toplantı için şu üç maddeyi bitirmemiz gerekiyor:

• Ahmet Mehmet'i arasın
• Mehmet Berfin'i arasın
• Berfin de beni arasın"

Input: "Let's meet at 2 no wait actually 3 would be better"
Output: "Let's meet at 3."

Input: "I need to buy milk eggs bread and also pick up the dry cleaning"
Output: "I need to buy:
• Milk
• Eggs
• Bread

Also pick up the dry cleaning."

Template: {template_rules}
Context: {ctx}{dictionary_rules}{tone_rules}

Remember: Output ONLY the cleaned text. Start immediately with the text itself. When you see multiple tasks/items, FORMAT AS BULLET LIST.""".strip()

        return base_rules

    def _repair_system_prompt(*, base_system: str) -> str:
        return """DICTATION TEXT CLEANER - STRICT MODE

Previous output REJECTED for adding new words.

ABSOLUTE RULES:
1. Use ONLY words that appear in the INPUT
2. You may: remove words, reorder slightly, add punctuation/newlines
3. You may NOT: add new words, translate, paraphrase, explain
4. NO preambles. NO "Here is...". NO explanations.
5. Output STARTS immediately with the cleaned text.

ALLOWED:
• Remove fillers (um, uh, yani, işte, etc.)
• Remove stutters/repetitions
• Fix punctuation
• Handle self-corrections (keep final version)
• Format lists if numbered

OUTPUT FORMAT: Just the cleaned text, nothing else. Start with the first word of the cleaned text."""

    def _build_prompt(*, system: str, user_text: str, is_email: bool = False) -> str:
        if tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        # Add explicit instruction in user message to prevent "helpful" preambles
        if is_email:
            user_message = f"Clean this dictation transcript. If multiple tasks/items, format as bullet list. Output ONLY the cleaned text, no preambles:\n\n{user_text}"
        else:
            user_message = f"Clean this dictation transcript. Output ONLY the cleaned text, no preambles:\n\n{user_text}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _parse_format_context(ctx: object) -> tuple[str, Optional[str], Optional[str], Optional[list], Optional[list], Optional[dict]]:
        active_app = ""
        domain: Optional[str] = None
        title: Optional[str] = None
        dictionary: Optional[list] = None
        snippets: Optional[list] = None
        tone: Optional[dict] = None

        if isinstance(ctx, dict):
            aa = ctx.get("active_app")
            if isinstance(aa, str):
                active_app = aa

            browser = ctx.get("browser")
            if isinstance(browser, dict):
                d = browser.get("domain")
                if isinstance(d, str) and d.strip():
                    domain = d.strip()
                t = browser.get("title")
                if isinstance(t, str) and t.strip():
                    title = t.strip()
            
            # Parse tone for style guidance
            tone_data = ctx.get("tone")
            if isinstance(tone_data, dict):
                tone_name = tone_data.get("name")
                tone_modifier = tone_data.get("promptModifier")
                if isinstance(tone_name, str) and tone_name.strip():
                    tone = {
                        "name": tone_name.strip(),
                        "promptModifier": tone_modifier.strip() if isinstance(tone_modifier, str) else "",
                    }
            
            # Parse dictionary entries for pronunciation/spelling guidance
            dict_entries = ctx.get("dictionary")
            if isinstance(dict_entries, list) and dict_entries:
                dictionary = [
                    {"term": e.get("term"), "pronunciation": e.get("pronunciation"), "category": e.get("category")}
                    for e in dict_entries
                    if isinstance(e, dict) and e.get("term")
                ]
            
            # Parse snippets for text expansion
            snippet_entries = ctx.get("snippets")
            if isinstance(snippet_entries, list) and snippet_entries:
                snippets = [
                    {"name": s.get("name"), "content": s.get("content"), "category": s.get("category")}
                    for s in snippet_entries
                    if isinstance(s, dict) and s.get("name") and s.get("content")
                ]

        return active_app, domain, title, dictionary, snippets, tone

    async def _format_text(
        *,
        text: str,
        template: str,
        language_hint: str,
        context: object,
    ) -> dict:
        req_t0 = time.perf_counter()

        raw = _normalize_ws(text)
        if not raw:
            return {
                "text": "",
                "raw_text": "",
                "llm_input_text": "",
                "llm_output_text": "",
                "selected_text": "",
                "preprocess": {},
                "template": "generic",
                "language": _infer_language(text="", hint=language_hint),
                "llm_ms": 0.0,
                "server_total_ms": 0.0,
                "prompt_version": FORMATTER_PROMPT_VERSION,
                "passes": [],
            }

        active_app, domain, title, dictionary, snippets, tone = _parse_format_context(context)
        if template == "auto":
            template = _infer_template(active_app=active_app, domain=domain, title=title)

        language = _infer_language(text=raw, hint=language_hint)
        
        # Apply snippet expansion BEFORE LLM processing
        expanded_text = raw
        snippet_expansions = []
        if snippets:
            for snippet in snippets:
                name = snippet.get("name", "").lower()
                content = snippet.get("content", "")
                if name and content:
                    # Check if snippet name appears in text (case-insensitive)
                    import re
                    pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
                    if pattern.search(expanded_text):
                        expanded_text = pattern.sub(content, expanded_text)
                        snippet_expansions.append({"name": name, "content": content})
        
        llm_input_text, preprocess_info = _preprocess_llm_input(expanded_text)
        llm_input_text = _normalize_ws(llm_input_text)
        
        # Add snippet expansion info to preprocess
        if snippet_expansions:
            preprocess_info["snippet_expansions"] = snippet_expansions

        if not llm or not sampling:
            return {
                "text": llm_input_text,
                "raw_text": raw,
                "llm_input_text": llm_input_text,
                "llm_output_text": llm_input_text,
                "selected_text": llm_input_text,
                "preprocess": preprocess_info,
                "template": template,
                "language": language,
                "llm_ms": 0.0,
                "server_total_ms": (time.perf_counter() - req_t0) * 1000,
                "fallback": "no_llm",
                "error": llm_init_error,
                "prompt_version": FORMATTER_PROMPT_VERSION,
                "passes": [],
            }

        system = _system_prompt(
            template=template,
            active_app=active_app,
            domain=domain,
            title=title,
            language=language,
            dictionary=dictionary,
            tone=tone,
        )
        is_email_template = template == "email"
        prompt = _build_prompt(system=system, user_text=llm_input_text, is_email=is_email_template)

        passes: list[dict] = []

        t0 = time.perf_counter()
        async with gen_lock:
            out = await asyncio.to_thread(llm.generate, prompt, sampling)
        llm_ms = (time.perf_counter() - t0) * 1000
        server_total_ms = (time.perf_counter() - req_t0) * 1000

        llm_output_1_raw = (out.outputs[0].text or "").strip()
        llm_output_1 = _trim_llm_preamble(llm_output_1_raw)
        
        # Allow reordering for email/doc templates (formatting contexts)
        allow_reordering = template in {"email", "doc"}
        
        # Tones that allow more LLM freedom (word addition/removal)
        tone_name = tone.get("name", "") if tone else ""
        skip_safety_for_tone = tone_name in {"Direct", "Verbose"}
        
        safety_1 = _safety_check(input_text=llm_input_text, output_text=llm_output_1, allow_reordering=allow_reordering, dictionary_terms=dictionary)
        passes.append(
            {
                "pass": "main",
                "output_text": llm_output_1,
                "safety": safety_1,
            }
        )

        # For Direct/Verbose tones, trust LLM output without strict safety check
        if safety_1.get("ok") or skip_safety_for_tone:
            return {
                "text": llm_output_1,
                "raw_text": raw,
                "llm_input_text": llm_input_text,
                "llm_output_text": llm_output_1,
                "selected_text": llm_output_1,
                "preprocess": preprocess_info,
                "template": template,
                "language": language,
                "llm_ms": llm_ms,
                "server_total_ms": server_total_ms,
                "prompt_version": FORMATTER_PROMPT_VERSION,
                "passes": passes,
                "used_pass": "main" if safety_1.get("ok") else "main_tone_bypass",
            }

        trimmed_1 = _trim_output_to_safe_candidate(input_text=llm_input_text, output_text=llm_output_1)
        if trimmed_1 is not None:
            safety_trim_1 = _safety_check(input_text=llm_input_text, output_text=trimmed_1, allow_reordering=allow_reordering, dictionary_terms=dictionary)
            passes.append(
                {
                    "pass": "main_trim",
                    "output_text": trimmed_1,
                    "safety": safety_trim_1,
                }
            )
            return {
                "text": trimmed_1,
                "raw_text": raw,
                "llm_input_text": llm_input_text,
                "llm_output_text": llm_output_1,
                "selected_text": trimmed_1,
                "preprocess": preprocess_info,
                "template": template,
                "language": language,
                "llm_ms": llm_ms,
                "server_total_ms": server_total_ms,
                "prompt_version": FORMATTER_PROMPT_VERSION,
                "passes": passes,
                "used_pass": "main_trim",
            }

        # Repair pass (still uses the LLM, but enforces "no new words" more aggressively).
        repair_system = _repair_system_prompt(base_system=system)
        repair_prompt = _build_prompt(system=repair_system, user_text=llm_input_text, is_email=is_email_template)

        t1 = time.perf_counter()
        async with gen_lock:
            out2 = await asyncio.to_thread(llm.generate, repair_prompt, sampling)
        llm_ms += (time.perf_counter() - t1) * 1000
        server_total_ms = (time.perf_counter() - req_t0) * 1000

        llm_output_2_raw = (out2.outputs[0].text or "").strip()
        llm_output_2 = _trim_llm_preamble(llm_output_2_raw)
        safety_2 = _safety_check(input_text=llm_input_text, output_text=llm_output_2, allow_reordering=allow_reordering, dictionary_terms=dictionary)
        passes.append(
            {
                "pass": "repair",
                "output_text": llm_output_2,
                "safety": safety_2,
            }
        )

        if safety_2.get("ok"):
            return {
                "text": llm_output_2,
                "raw_text": raw,
                "llm_input_text": llm_input_text,
                "llm_output_text": llm_output_2,
                "selected_text": llm_output_2,
                "preprocess": preprocess_info,
                "template": template,
                "language": language,
                "llm_ms": llm_ms,
                "server_total_ms": server_total_ms,
                "prompt_version": FORMATTER_PROMPT_VERSION,
                "passes": passes,
                "used_pass": "repair",
            }

        trimmed_2 = _trim_output_to_safe_candidate(input_text=llm_input_text, output_text=llm_output_2)
        if trimmed_2 is not None:
            safety_trim_2 = _safety_check(input_text=llm_input_text, output_text=trimmed_2, allow_reordering=allow_reordering, dictionary_terms=dictionary)
            passes.append(
                {
                    "pass": "repair_trim",
                    "output_text": trimmed_2,
                    "safety": safety_trim_2,
                }
            )
            return {
                "text": trimmed_2,
                "raw_text": raw,
                "llm_input_text": llm_input_text,
                "llm_output_text": llm_output_2,
                "selected_text": trimmed_2,
                "preprocess": preprocess_info,
                "template": template,
                "language": language,
                "llm_ms": llm_ms,
                "server_total_ms": server_total_ms,
                "prompt_version": FORMATTER_PROMPT_VERSION,
                "passes": passes,
                "used_pass": "repair_trim",
            }

        return {
            "text": llm_input_text,
            "raw_text": raw,
            "llm_input_text": llm_input_text,
            "llm_output_text": llm_output_2 or llm_output_1,
            "selected_text": llm_input_text,
            "preprocess": preprocess_info,
            "template": template,
            "language": language,
            "llm_ms": llm_ms,
            "server_total_ms": server_total_ms,
            "prompt_version": FORMATTER_PROMPT_VERSION,
            "passes": passes,
            "used_pass": "fallback_input",
            "fallback": "unsafe_llm_output",
        }

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
        max_new_tokens = 448  # Base value, will be dynamically adjusted per audio duration
        initial_prompt = ""  # Custom vocabulary from dictionary
        format_enabled = True
        format_template = "auto"
        format_context: object = {}
        format_language_hint = "auto"

        # Push-to-talk: transcribe on fn key release for best accuracy
        # For 30s+ audio, use overlap chunking to prevent word loss
        segment_silence_ms = 0
        min_segment_ms = 200
        max_audio_sec = 26  # Chunk at 26s with 4s overlap for 30s+ audio
        overlap_sec = 4  # Keep last 4s for next chunk to prevent word loss
        
        last_voice_ts: Optional[float] = None
        flush_lock = asyncio.Lock()
        session_epoch = 0
        overlap_buffer = bytearray()  # Store last 4s for overlap

        async def flush_segment(reason: str, *, used_overlap: bool = False) -> Optional[dict]:
            nonlocal buffer, segments, last_voice_ts, session_epoch, overlap_buffer
            async with flush_lock:
                if not buffer:
                    print(f"⚠️  flush_segment called but buffer is empty (reason: {reason})")
                    return None
                if fmt != "pcm_s16le":
                    await ws.send_json({"type": "error", "error": f"Unsupported fmt: {fmt}"})
                    buffer = bytearray()
                    return None

                audio_sec = (len(buffer) / 2) / float(sr)
                
                print(f"🎤 Transcribing segment: {audio_sec:.2f}s ({reason})")

                epoch = session_epoch
                pcm = bytes(buffer)
                
                # For overlap chunking: save last `overlap_sec` seconds for next chunk
                if reason == 'max_len':
                    overlap_bytes = int(overlap_sec * sr * 2)
                    if len(buffer) > overlap_bytes:
                        overlap_buffer = bytearray(buffer[-overlap_bytes:])
                        print(f"💾 Saved {overlap_sec}s overlap for next chunk")
                    buffer = bytearray()
                    last_voice_ts = time.monotonic()
                elif reason in ('end', 'reset'):
                    buffer = bytearray()
                    overlap_buffer = bytearray()
                    last_voice_ts = None
                else:
                    buffer = bytearray()
                    last_voice_ts = time.monotonic()

                # VAD check: skip silent audio to avoid garbage output
                if not _has_voice_activity(pcm, sr=sr):
                    print(f"🔇 Skipping silent audio segment ({audio_sec:.2f}s)")
                    return {
                        "reason": reason,
                        "text": "",
                        "full_text": " ".join(s for s in segments if s).strip(),
                        "overlap": {"used": bool(used_overlap), "removed_words": 0, "sec": overlap_sec if used_overlap else 0},
                        "audio_sec": audio_sec,
                        "rtf_cuda": None,
                        "timings_ms": {"skipped": "silence"},
                    }

                # Dynamic max_new_tokens based on audio duration (clamped to engine capacity)
                estimated_tokens = int(audio_sec * 4 * 1.5)
                dynamic_max_tokens = max(max_new_tokens, min(estimated_tokens, engine_max_output_len))

                text_prefix_with_context = _build_text_prefix(language=language, task=task, timestamps=timestamps)
                
                cfg_local = InferenceConfig(
                    text_prefix=text_prefix_with_context,
                    initial_prompt=initial_prompt,  # Custom vocabulary from dictionary
                    num_beams=num_beams,
                    max_new_tokens=dynamic_max_tokens,
                )

                t0 = time.perf_counter()
                result = await asyncio.to_thread(
                    runner.transcribe_pcm16le_with_timings, pcm, sr=sr, cfg=cfg_local
                )

                if epoch != session_epoch:
                    return None

                server_total_ms = (time.perf_counter() - t0) * 1000
                text = (result.get("text") or "").strip()
                
                # Filter out garbage patterns that indicate silence or unintelligible audio
                # Whisper sometimes outputs `||_|...` or similar patterns for silent/noisy audio
                garbage_chars = set(text)
                if garbage_chars and garbage_chars <= {'|', '_', ' ', '\n'}:
                    print(f"⚠️  Filtered garbage output: '{text[:50]}...' (likely silence)")
                    text = ""
                elif '||_|' in text or '|_|_|' in text:
                    # Remove garbage suffix if present
                    import re
                    garbage_pattern = re.compile(r'\|[\|_\s]+$')
                    cleaned = garbage_pattern.sub('', text).strip()
                    if cleaned != text:
                        print(f"⚠️  Trimmed garbage suffix from: '{text[-50:]}' -> '{cleaned[-50:]}'")
                        text = cleaned
                
                print(f"✅ Transcription result: '{text}' ({server_total_ms:.0f}ms)")
                
                overlap_removed_words = 0
                if text:
                    # If we used overlap audio, this chunk starts with already-transcribed content.
                    # Prefer the newer chunk's wording by trimming the overlapping suffix from the
                    # previous segment, then appending the new text.
                    if used_overlap and segments:
                        prev = segments[-1]
                        overlap_removed_words, cut_idx = _find_overlap_suffix_prefix(
                            prev, text, language=language, min_words=6, max_words=80
                        )
                        if overlap_removed_words > 0 and cut_idx >= 0:
                            trimmed_prev = prev[:cut_idx].rstrip()
                            if trimmed_prev:
                                segments[-1] = trimmed_prev
                            else:
                                segments.pop()
                            print(f"🧹 Removed ~{overlap_removed_words} overlapped words from previous segment")

                    segments.append(text)

                full_text = " ".join(s for s in segments if s).strip()
                timings_ms = dict(result.get("timings_ms") or {})
                timings_ms["server_total"] = server_total_ms
                timings_ms["max_tokens_used"] = dynamic_max_tokens
                if "total" in timings_ms:
                    timings_ms["server_overhead"] = max(0.0, server_total_ms - float(timings_ms["total"]))

                return {
                    "reason": reason,
                    "text": text,
                    "full_text": full_text,
                    "overlap": {
                        "used": bool(used_overlap),
                        "removed_words": overlap_removed_words,
                        "sec": overlap_sec if used_overlap else 0,
                    },
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
                    payload = await flush_segment("silence", used_overlap=False)
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
                    "ready": runner is not None,
                    "whisper_engine": runner is not None,
                    "formatter_engine": llm is not None,
                    "errors": {"whisper": runner_error, "formatter": llm_init_error},
                }
            )

            while True:
                try:
                    msg = await ws.receive()
                except RuntimeError as e:
                    # Client disconnected
                    print(f"WebSocket receive error: {e}")
                    break
                    
                if msg.get("text") is not None:
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        await ws.send_json({"type": "error", "error": "Invalid JSON message"})
                        continue

                    mtype = data.get("type")
                    if mtype == "start":
                        if runner is None:
                            await ws.send_json({"type": "error", "error": runner_error or "Whisper engine not ready"})
                            active = False
                            continue

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
                        if max_new_tokens <= 0:
                            max_new_tokens = 1
                        max_new_tokens = min(max_new_tokens, engine_max_output_len)

                        format_enabled = bool(data.get("format", format_enabled))
                        format_template = str(data.get("format_template", data.get("template", format_template)))
                        format_context = data.get("format_context", data.get("context", format_context))
                        format_language_hint = str(data.get("format_language", language))

                        segment_silence_ms = int(data.get("segment_silence_ms", segment_silence_ms))
                        min_segment_ms = int(data.get("min_segment_ms", min_segment_ms))
                        
                        # Build initial_prompt from dictionary entries for Whisper
                        # Limit: ~50 chars per term, max 100 chars total to stay under token limit
                        # Engine limit is 466 tokens, max_new_tokens=448, so we have ~18 tokens for prompt
                        # ~4 chars per token means ~70 chars max, we use 100 to be safe
                        initial_prompt = ""
                        if isinstance(format_context, dict):
                            dictionary = format_context.get("dictionary")
                            if isinstance(dictionary, list) and dictionary:
                                # Extract terms for Whisper's initial prompt (spelling guidance)
                                terms = []
                                total_len = 0
                                max_total_chars = 60  # Safe limit for token budget
                                max_term_chars = 30   # Skip very long terms
                                for entry in dictionary[:30]:  # Limit to 30 entries
                                    if isinstance(entry, dict):
                                        term = entry.get("term", "")
                                        if term and isinstance(term, str):
                                            term = term.strip()
                                            # Skip terms that are too long
                                            if len(term) > max_term_chars:
                                                continue
                                            # Check if adding this term would exceed limit
                                            add_len = len(term) + 2  # +2 for ", "
                                            if total_len + add_len > max_total_chars:
                                                break
                                            terms.append(term)
                                            total_len += add_len
                                if terms:
                                    initial_prompt = " " + ", ".join(terms) + "."
                                    print(f"📚 Dictionary terms for Whisper: {initial_prompt.strip()}")

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

                        print(f"🎬 Session started: lang={language}, sr={sr}, fmt={fmt}")
                        await ws.send_json(
                            {
                                "type": "started",
                                "sr": sr,
                                "fmt": fmt,
                                "segments": segments,
                            }
                        )
                    elif mtype == "end":
                        last = None
                        if active:
                            # Wait briefly for any final audio chunks to arrive
                            await asyncio.sleep(0.05)  # 50ms grace period
                            
                            print(f"📥 End event received, buffer size: {len(buffer)} bytes ({len(buffer)/2/sr:.2f}s)")
                            used_overlap = False
                            if overlap_buffer and buffer:
                                buffer = overlap_buffer + buffer
                                overlap_buffer = bytearray()
                                used_overlap = True
                                print(f"🔄 Prepended {overlap_sec}s overlap to final chunk")
                            last = await flush_segment("end", used_overlap=used_overlap)
                            
                            # If we got a result from the final segment, send it as a segment first
                            if last and last.get("text"):
                                await ws.send_json({
                                    "type": "segment",
                                    "text": last["text"],
                                    "full_text": last["full_text"],
                                    "reason": "end",
                                    "audio_sec": last.get("audio_sec"),
                                    "overlap": last.get("overlap"),
                                    "timings_ms": last.get("timings_ms"),
                                    "rtf_cuda": last.get("rtf_cuda"),
                                })
                                print(f"📦 Final segment: '{last['text']}' ({last.get('audio_sec', 0):.1f}s, end)")
                        
                        active = False
                        full_text = " ".join(s for s in segments if s).strip()
                        formatted_text = full_text
                        format_out: Optional[dict] = None
                        if format_enabled and full_text:
                            try:
                                format_out = await _format_text(
                                    text=full_text,
                                    template=format_template,
                                    language_hint=format_language_hint,
                                    context=format_context,
                                )
                                maybe = (format_out.get("text") if isinstance(format_out, dict) else None) or ""
                                if isinstance(maybe, str) and maybe.strip():
                                    formatted_text = maybe.strip()
                            except Exception as e:
                                format_out = {"error": repr(e)}

                        await ws.send_json(
                            {
                                "type": "final",
                                "text": full_text,
                                "formatted_text": formatted_text,
                                "format": format_out,
                                "segments": segments,
                                "last": last,
                            }
                        )
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

                    # Smart overlap chunking for 30s+ audio
                    # Chunk at `max_audio_sec`, keep `overlap_sec` for next chunk to prevent word loss
                    max_bytes = sr * max_audio_sec * 2
                    if len(buffer) + len(chunk) > max_bytes:
                        # Prepend overlap from previous chunk
                        used_overlap = False
                        if overlap_buffer:
                            buffer = overlap_buffer + buffer
                            overlap_buffer = bytearray()
                            print(f"🔄 Prepended {overlap_sec}s overlap to current chunk")
                            used_overlap = True
                        
                        payload = await flush_segment("max_len", used_overlap=used_overlap)
                        if payload:
                            await ws.send_json({"type": "segment", **payload})
                            print(f"📦 Overlap chunk: '{payload['text']}'")

                    buffer.extend(chunk)
                    last_voice_ts = time.monotonic()
                    
                    # Debug: Log buffer size periodically
                    if len(buffer) % 160000 == 0:  # Every ~5 seconds
                        print(f"📊 Buffer: {len(buffer)/2/sr:.1f}s")
                elif msg.get("type") == "websocket.disconnect":
                    print("Client disconnected")
                    break
        except WebSocketDisconnect:
            print("WebSocket disconnected gracefully")
            return
        except Exception as e:
            print(f"WebSocket error: {e}")
            import traceback
            traceback.print_exc()
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
        max_new_tokens: int = 448,
        timings: bool = False,
    ):
        if runner is None:
            raise HTTPException(status_code=503, detail=runner_error or "Whisper engine not ready")

        req_t0 = time.perf_counter()
        audio_bytes = await file.read()
        suffix = Path(file.filename or "").suffix or ".wav"
        text_prefix = "<|startoftranscript|>"
        if language:
            text_prefix += f"<|{language}|>"
        text_prefix += f"<|{task}|>"
        if not timestamps:
            text_prefix += "<|notimestamps|>"

        # Dynamic max_new_tokens based on file size estimate
        # Rough estimate: 16kHz * 2 bytes = 32KB per second
        estimated_audio_sec = len(audio_bytes) / 32000
        estimated_tokens = int(estimated_audio_sec * 4 * 1.5)
        dynamic_max_tokens = min(engine_max_output_len, max(max_new_tokens, estimated_tokens))

        cfg = InferenceConfig(text_prefix=text_prefix, num_beams=num_beams, max_new_tokens=dynamic_max_tokens)
        if timings:
            result = runner.transcribe_bytes_with_timings(audio_bytes, suffix=suffix, cfg=cfg)
            server_total_ms = (time.perf_counter() - req_t0) * 1000
            timings_ms = result.get("timings_ms", {})
            timings_ms["server_total"] = server_total_ms
            timings_ms["max_tokens_used"] = dynamic_max_tokens
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
