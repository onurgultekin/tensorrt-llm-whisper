from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .assets import ensure_whisper_assets, ensure_whisper_checkpoint


@dataclass(frozen=True)
class BuildConfig:
    model_name: str = "large-v3"
    inference_precision: str = "float16"
    use_weight_only: bool = True
    weight_only_precision: str = "int8"  # int8|int4
    max_batch_size: int = 1
    max_beam_width: int = 1
    max_encoder_input_len: int = 3000
    max_output_len: int = 96
    paged_kv_cache: bool = True
    remove_input_padding: bool = True

    def build_id(self) -> str:
        wo = f"wo_{self.weight_only_precision}" if self.use_weight_only else "fp16"
        flags = []
        if self.paged_kv_cache:
            flags.append("pagedkv")
        if self.remove_input_padding:
            flags.append("rip")
        flags_s = ("_" + "_".join(flags)) if flags else ""
        return f"{self.model_name}_{wo}_mb{self.max_batch_size}_bw{self.max_beam_width}{flags_s}"


def _env_for_subprocess() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return env


def convert_checkpoint(*, assets_dir: Path, checkpoints_dir: Path, cfg: BuildConfig) -> Path:
    ensure_whisper_assets(assets_dir)
    ensure_whisper_checkpoint(assets_dir, cfg.model_name)

    out_dir = checkpoints_dir / cfg.build_id()
    encoder_st = out_dir / "encoder" / "rank0.safetensors"
    decoder_st = out_dir / "decoder" / "rank0.safetensors"
    if encoder_st.exists() and decoder_st.exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        "-m",
        "trtllm_whisper.convert_checkpoint",
        "--model_dir",
        str(assets_dir),
        "--model_name",
        cfg.model_name,
        "--dtype",
        cfg.inference_precision,
        "--logits_dtype",
        cfg.inference_precision,
        "--output_dir",
        str(out_dir),
    ]
    if cfg.use_weight_only:
        cmd += ["--use_weight_only", "--weight_only_precision", cfg.weight_only_precision]

    subprocess.run(cmd, check=True, env=_env_for_subprocess())
    if not (encoder_st.exists() and decoder_st.exists()):
        raise RuntimeError("Checkpoint conversion finished, but expected safetensors not found.")
    return out_dir


def build_engines(*, checkpoint_dir: Path, engines_dir: Path, cfg: BuildConfig) -> Path:
    engine_dir = engines_dir / cfg.build_id()
    enc_engine = engine_dir / "encoder" / "rank0.engine"
    dec_engine = engine_dir / "decoder" / "rank0.engine"
    if enc_engine.exists() and dec_engine.exists():
        return engine_dir

    (engine_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (engine_dir / "decoder").mkdir(parents=True, exist_ok=True)

    paged = "enable" if cfg.paged_kv_cache else "disable"
    rip = "enable" if cfg.remove_input_padding else "disable"

    # Encoder
    enc_cmd = [
        "trtllm-build",
        "--checkpoint_dir",
        str(checkpoint_dir / "encoder"),
        "--output_dir",
        str(engine_dir / "encoder"),
        "--moe_plugin",
        "disable",
        "--max_batch_size",
        str(cfg.max_batch_size),
        "--gemm_plugin",
        "disable",
        "--bert_attention_plugin",
        cfg.inference_precision,
        "--max_input_len",
        str(cfg.max_encoder_input_len),
        "--max_seq_len",
        str(cfg.max_encoder_input_len),
        "--remove_input_padding",
        rip,
    ]

    # Decoder
    # max_seq_len needs to account for prompt + generated tokens; upstream uses 114 (14 + 100).
    dec_max_seq_len = 14 + cfg.max_output_len + 4
    dec_cmd = [
        "trtllm-build",
        "--checkpoint_dir",
        str(checkpoint_dir / "decoder"),
        "--output_dir",
        str(engine_dir / "decoder"),
        "--moe_plugin",
        "disable",
        "--max_beam_width",
        str(cfg.max_beam_width),
        "--max_batch_size",
        str(cfg.max_batch_size),
        "--max_seq_len",
        str(dec_max_seq_len),
        "--max_input_len",
        "14",
        "--max_encoder_input_len",
        str(cfg.max_encoder_input_len),
        "--gemm_plugin",
        cfg.inference_precision,
        "--bert_attention_plugin",
        cfg.inference_precision,
        "--gpt_attention_plugin",
        cfg.inference_precision,
        "--paged_kv_cache",
        paged,
        "--remove_input_padding",
        rip,
    ]

    subprocess.run(enc_cmd, check=True, env=_env_for_subprocess())
    subprocess.run(dec_cmd, check=True, env=_env_for_subprocess())
    if not (enc_engine.exists() and dec_engine.exists()):
        raise RuntimeError("Engine build finished, but expected rank0.engine files not found.")
    return engine_dir

