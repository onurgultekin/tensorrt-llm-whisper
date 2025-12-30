from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm.bindings import GptJsonConfig
from tensorrt_llm.runtime import PYTHON_BINDINGS

from .tokenizer import get_tokenizer
from .whisper_utils import SAMPLE_RATE, load_audio, load_audio_wav_format, log_mel_spectrogram

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def _read_component_config(engine_dir: Path, component: str) -> dict:
    cfg_path = engine_dir / component / "config.json"
    with cfg_path.open("r") as f:
        config = json.load(f)
    merged = {}
    merged.update(config.get("pretrained_config", {}))
    merged.update(config.get("build_config", {}))
    return merged


@dataclass(frozen=True)
class InferenceConfig:
    max_new_tokens: int = 448  # Increased for longer transcriptions (~350 words)
    num_beams: int = 1
    text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    initial_prompt: str = ""  # Custom vocabulary/context to guide transcription
    padding_strategy: str = "max"  # max|longest (single file behaves same) | nopad (cpp only)


class WhisperTRTLLMRunner:
    def __init__(
        self,
        *,
        engine_dir: Path,
        assets_dir: Path,
        max_batch_size: int,
        max_output_len: int,
        max_beam_width: int,
    ):
        if not PYTHON_BINDINGS:
            raise RuntimeError(
                "tensorrt_llm python bindings (C++ runtime) not available. "
                "Make sure you installed a TensorRT-LLM build with PYTHON_BINDINGS enabled."
            )

        self.engine_dir = engine_dir
        self.assets_dir = assets_dir
        self.max_output_len = max(1, int(max_output_len))

        enc_cfg = _read_component_config(engine_dir, "encoder")
        dec_cfg = _read_component_config(engine_dir, "decoder")

        self.n_mels = int(enc_cfg["n_mels"])
        self.num_languages = int(enc_cfg.get("num_languages", 99))
        vocab_size = int(dec_cfg["vocab_size"])

        tokenizer_name = "multilingual" if vocab_size >= 51865 else "gpt2"
        tok_path = assets_dir / f"{tokenizer_name}.tiktoken"
        if not tok_path.exists():
            raise FileNotFoundError(f"Missing tokenizer vocab: {tok_path}")
        self.tokenizer = get_tokenizer(
            name=tokenizer_name,
            num_languages=self.num_languages,
            tokenizer_dir=str(assets_dir),
        )
        self.eot_id = self.tokenizer.encode("<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set)[0]

        # Ensure decoder engine supports inflight batching (C++ runtime path).
        json_config = GptJsonConfig.parse_file(engine_dir / "decoder" / "config.json")
        if not json_config.model_config.supports_inflight_batching:
            raise RuntimeError(
                "Decoder engine does not support inflight batching; rebuild with "
                "--paged_kv_cache enable and --remove_input_padding enable."
            )

        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        kv_cache_fraction = float(os.getenv("VOICEOS_WHISPER_KV_CACHE_FRACTION", "0.3"))
        if not (0.01 <= kv_cache_fraction <= 0.95):
            raise ValueError(
                "VOICEOS_WHISPER_KV_CACHE_FRACTION must be between 0.01 and 0.95 "
                f"(got {kv_cache_fraction})"
            )
        cross_kv_cache_fraction = float(os.getenv("VOICEOS_WHISPER_CROSS_KV_CACHE_FRACTION", "0.5"))
        if not (0.0 <= cross_kv_cache_fraction <= 1.0):
            raise ValueError(
                "VOICEOS_WHISPER_CROSS_KV_CACHE_FRACTION must be between 0.0 and 1.0 "
                f"(got {cross_kv_cache_fraction})"
            )

        self.model_runner = ModelRunnerCpp.from_dir(
            engine_dir=engine_dir,
            is_enc_dec=True,
            max_batch_size=max_batch_size,
            max_input_len=3000,
            max_output_len=self.max_output_len,
            max_beam_width=max_beam_width,
            debug_mode=False,
            kv_cache_free_gpu_memory_fraction=kv_cache_fraction,
            cross_kv_cache_fraction=cross_kv_cache_fraction,
        )

    def _decode_tokens(self, token_ids: list[int]) -> str:
        text = self.tokenizer.decode(token_ids).strip()
        # Remove Whisper special tokens in decoded text.
        text = re.sub(r"<\|.*?\|>", "", text).strip()
        return text

    def transcribe_file(self, audio_path: Path, cfg: Optional[InferenceConfig] = None) -> str:
        cfg = cfg or InferenceConfig()
        mel = log_mel_spectrogram(
            str(audio_path),
            self.n_mels,
            device="cuda",
            mel_filters_dir=str(self.assets_dir),
        ).type(torch.float16)

        mel = mel.unsqueeze(0)  # [B=1, n_mels, frames]
        if cfg.padding_strategy == "max":
            if mel.shape[2] < 3000:
                mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[2]))
        elif cfg.padding_strategy == "longest":
            pass
        elif cfg.padding_strategy == "nopad":
            pass
        else:
            raise ValueError("padding_strategy must be one of: max|longest|nopad")

        input_lengths = torch.tensor([mel.shape[2]], dtype=torch.int32, device="cuda")
        
        # Build full prompt: initial_prompt (with <|startofprev|>) + text_prefix
        full_prefix = cfg.text_prefix
        if cfg.initial_prompt:
            clean_prompt = cfg.initial_prompt.strip()
            full_prefix = f"<|startofprev|>{clean_prompt}<|endoftext|>" + cfg.text_prefix
        
        prompt_ids = self.tokenizer.encode(full_prefix, allowed_special=self.tokenizer.special_tokens_set)
        decoder_input_ids = torch.tensor(prompt_ids, dtype=torch.int64).unsqueeze(0)  # [B=1, prompt]

        # ModelRunnerCpp expects encoder_input_features as [B, T, n_mels]
        encoder_input_features = mel.transpose(1, 2)

        with torch.no_grad():
            outputs = self.model_runner.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=encoder_input_features,
                encoder_output_lengths=input_lengths // 2,
                max_new_tokens=min(cfg.max_new_tokens, self.max_output_len),
                end_id=self.eot_id,
                pad_id=self.eot_id,
                num_beams=cfg.num_beams,
                output_sequence_lengths=True,
                return_dict=True,
            )

        out = outputs["output_ids"].cpu().numpy().tolist()
        token_ids = out[0][0]
        token_ids = token_ids[len(prompt_ids):]
        if self.eot_id in token_ids:
            token_ids = token_ids[:token_ids.index(self.eot_id)]
        return self._decode_tokens(token_ids)

    def transcribe_file_with_timings(
        self, audio_path: Path, *, cfg: Optional[InferenceConfig] = None
    ) -> dict[str, Any]:
        cfg = cfg or InferenceConfig()
        total_t0 = time.perf_counter()

        audio_decode_t0 = time.perf_counter()
        if str(audio_path).endswith(".wav"):
            audio, _ = load_audio_wav_format(str(audio_path))
        else:
            audio = load_audio(str(audio_path))
        audio_decode_ms = (time.perf_counter() - audio_decode_t0) * 1000

        mel_t0 = time.perf_counter()
        mel = log_mel_spectrogram(
            audio,
            self.n_mels,
            device="cuda",
            mel_filters_dir=str(self.assets_dir),
        ).type(torch.float16)
        torch.cuda.synchronize()
        mel_ms = (time.perf_counter() - mel_t0) * 1000

        mel = mel.unsqueeze(0)  # [B=1, n_mels, frames]
        
        pad_t0 = time.perf_counter()
        if cfg.padding_strategy == "max":
            if mel.shape[2] < 3000:
                mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[2]))
        elif cfg.padding_strategy == "longest":
            pass
        elif cfg.padding_strategy == "nopad":
            pass
        else:
            raise ValueError("padding_strategy must be one of: max|longest|nopad")
        torch.cuda.synchronize()
        pad_ms = (time.perf_counter() - pad_t0) * 1000

        # encoder_output_lengths must match the actual encoder output shape (after padding and downsampling)
        # Whisper encoder downsamples by 2x, so output length = padded_frames // 2
        input_lengths = torch.tensor([mel.shape[2] // 2], dtype=torch.int32, device="cuda")
        
        # Build full prompt: initial_prompt (with <|startofprev|>) + text_prefix
        full_prefix = cfg.text_prefix
        if cfg.initial_prompt:
            clean_prompt = cfg.initial_prompt.strip()
            full_prefix = f"<|startofprev|>{clean_prompt}<|endoftext|>" + cfg.text_prefix
        
        prompt_ids = self.tokenizer.encode(full_prefix, allowed_special=self.tokenizer.special_tokens_set)
        decoder_input_ids = torch.tensor(prompt_ids, dtype=torch.int64).unsqueeze(0)  # [B=1, prompt]

        # ModelRunnerCpp expects encoder_input_features as [B, T, n_mels]
        encoder_input_features = mel.transpose(1, 2)

        gen_wall_t0 = time.perf_counter()
        gen_evt_start = torch.cuda.Event(enable_timing=True)
        gen_evt_end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            gen_evt_start.record()
            outputs = self.model_runner.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=encoder_input_features,
                encoder_output_lengths=input_lengths,
                max_new_tokens=min(cfg.max_new_tokens, self.max_output_len),
                end_id=self.eot_id,
                pad_id=self.eot_id,
                num_beams=cfg.num_beams,
                output_sequence_lengths=True,
                return_dict=True,
            )
            gen_evt_end.record()
            torch.cuda.synchronize()
        gen_wall_ms = (time.perf_counter() - gen_wall_t0) * 1000
        gen_cuda_ms = float(gen_evt_start.elapsed_time(gen_evt_end))

        # output_ids: [B, beam, seq]
        post_t0 = time.perf_counter()
        out = outputs["output_ids"].cpu().numpy().tolist()
        token_ids = out[0][0]
        token_ids = token_ids[len(prompt_ids):]
        if self.eot_id in token_ids:
            token_ids = token_ids[:token_ids.index(self.eot_id)]
        text = self._decode_tokens(token_ids)
        post_ms = (time.perf_counter() - post_t0) * 1000

        total_ms = (time.perf_counter() - total_t0) * 1000
        return {
            "text": text,
            "timings_ms": {
                "total": total_ms,
                "audio_decode": audio_decode_ms,
                "mel": mel_ms,
                "pad": pad_ms,
                "generate_wall": gen_wall_ms,
                "generate_cuda": gen_cuda_ms,
                "postprocess": post_ms,
            },
        }

    def transcribe_bytes(self, audio_bytes: bytes, *, suffix: str = ".wav", cfg: Optional[InferenceConfig] = None) -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
            f.write(audio_bytes)
            f.flush()
            return self.transcribe_file(Path(f.name), cfg=cfg)

    def transcribe_bytes_with_timings(
        self, audio_bytes: bytes, *, suffix: str = ".wav", cfg: Optional[InferenceConfig] = None
    ) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
            f.write(audio_bytes)
            f.flush()
            return self.transcribe_file_with_timings(Path(f.name), cfg=cfg)

    def transcribe_pcm16le(self, pcm_bytes: bytes, *, sr: int = SAMPLE_RATE, cfg: Optional[InferenceConfig] = None) -> str:
        return self.transcribe_pcm16le_with_timings(pcm_bytes, sr=sr, cfg=cfg)["text"]

    def transcribe_pcm16le_with_timings(
        self, pcm_bytes: bytes, *, sr: int = SAMPLE_RATE, cfg: Optional[InferenceConfig] = None
    ) -> dict[str, Any]:
        cfg = cfg or InferenceConfig()
        if sr != SAMPLE_RATE:
            raise ValueError(f"Unsupported sample rate: {sr} (expected {SAMPLE_RATE})")

        total_t0 = time.perf_counter()

        decode_t0 = time.perf_counter()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_sec = float(audio.shape[0]) / float(sr) if audio.shape[0] else 0.0
        decode_ms = (time.perf_counter() - decode_t0) * 1000

        mel_t0 = time.perf_counter()
        mel = log_mel_spectrogram(
            audio,
            self.n_mels,
            device="cuda",
            mel_filters_dir=str(self.assets_dir),
        ).type(torch.float16)
        torch.cuda.synchronize()
        mel_ms = (time.perf_counter() - mel_t0) * 1000

        mel = mel.unsqueeze(0)  # [B=1, n_mels, frames]
        
        pad_t0 = time.perf_counter()
        if cfg.padding_strategy == "max":
            if mel.shape[2] < 3000:
                mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[2]))
        elif cfg.padding_strategy == "longest":
            pass
        elif cfg.padding_strategy == "nopad":
            pass
        else:
            raise ValueError("padding_strategy must be one of: max|longest|nopad")
        torch.cuda.synchronize()
        pad_ms = (time.perf_counter() - pad_t0) * 1000

        # encoder_output_lengths must match the actual encoder output shape (after padding and downsampling)
        # Whisper encoder downsamples by 2x, so output length = padded_frames // 2
        input_lengths = torch.tensor([mel.shape[2] // 2], dtype=torch.int32, device="cuda")
        
        # Build full prompt: initial_prompt (with <|startofprev|>) + text_prefix
        # Whisper expects: <|startofprev|> [context] <|startoftranscript|><|lang|><|task|><|notimestamps|>
        # The initial_prompt provides context/vocabulary hints BEFORE the main transcription tokens
        full_prefix = cfg.text_prefix
        if cfg.initial_prompt:
            clean_prompt = cfg.initial_prompt.strip()
            # Prepend with <|startofprev|> token for proper Whisper conditioning
            full_prefix = f"<|startofprev|>{clean_prompt}<|endoftext|>" + cfg.text_prefix
        
        prompt_ids = self.tokenizer.encode(full_prefix, allowed_special=self.tokenizer.special_tokens_set)
        decoder_input_ids = torch.tensor(prompt_ids, dtype=torch.int64).unsqueeze(0)  # [B=1, prompt]
        decoder_input_ids = torch.tensor(prompt_ids, dtype=torch.int64).unsqueeze(0)  # [B=1, prompt]

        encoder_input_features = mel.transpose(1, 2)  # [B, T, n_mels]

        gen_wall_t0 = time.perf_counter()
        gen_evt_start = torch.cuda.Event(enable_timing=True)
        gen_evt_end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            gen_evt_start.record()
            # Don't pass cross_attention_mask - let TensorRT-LLM handle it internally
            outputs = self.model_runner.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=encoder_input_features,
                encoder_output_lengths=input_lengths,
                max_new_tokens=min(cfg.max_new_tokens, self.max_output_len),
                end_id=self.eot_id,
                pad_id=self.eot_id,
                num_beams=cfg.num_beams,
                output_sequence_lengths=True,
                return_dict=True,
            )
            gen_evt_end.record()
            torch.cuda.synchronize()
        gen_wall_ms = (time.perf_counter() - gen_wall_t0) * 1000
        gen_cuda_ms = float(gen_evt_start.elapsed_time(gen_evt_end))

        post_t0 = time.perf_counter()
        out = outputs["output_ids"].cpu().numpy().tolist()
        token_ids = out[0][0]
        token_ids = token_ids[len(prompt_ids):]
        if self.eot_id in token_ids:
            token_ids = token_ids[:token_ids.index(self.eot_id)]
        text = self._decode_tokens(token_ids)
        text = self._decode_tokens(token_ids)
        post_ms = (time.perf_counter() - post_t0) * 1000

        total_ms = (time.perf_counter() - total_t0) * 1000
        rtf_cuda = (gen_cuda_ms / 1000.0) / audio_sec if audio_sec > 0 else None
        return {
            "text": text,
            "audio_sec": audio_sec,
            "rtf_cuda": rtf_cuda,
            "timings_ms": {
                "total": total_ms,
                "audio_decode": decode_ms,
                "mel": mel_ms,
                "pad": pad_ms,
                "generate_wall": gen_wall_ms,
                "generate_cuda": gen_cuda_ms,
                "postprocess": post_ms,
            },
        }
