from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


OPENAI_WHISPER_MODELS = {
    # Source of truth: https://github.com/openai/whisper/blob/main/whisper/__init__.py
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}

OPENAI_WHISPER_ASSETS = {
    "multilingual.tiktoken": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
    "gpt2.tiktoken": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken",
    "mel_filters.npz": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz",
}


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    reused: bool


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dst: Path, *, expected_sha256: str | None = None) -> DownloadResult:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.is_file():
        if expected_sha256 is None:
            return DownloadResult(dst, reused=True)
        if _sha256(dst) == expected_sha256:
            return DownloadResult(dst, reused=True)
        dst.unlink()

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    if expected_sha256 is not None:
        got = _sha256(tmp)
        if got != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch for {url}: expected {expected_sha256}, got {got}")

    os.replace(tmp, dst)
    return DownloadResult(dst, reused=False)


def ensure_whisper_assets(assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    for name, url in OPENAI_WHISPER_ASSETS.items():
        download_file(url, assets_dir / name)


def whisper_checkpoint_url(model_name: str) -> str:
    if model_name not in OPENAI_WHISPER_MODELS:
        raise ValueError(f"Unsupported model_name={model_name!r}. Known: {sorted(OPENAI_WHISPER_MODELS)}")
    return OPENAI_WHISPER_MODELS[model_name]


def ensure_whisper_checkpoint(assets_dir: Path, model_name: str) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    url = whisper_checkpoint_url(model_name)
    expected_sha256 = url.split("/")[-2]
    dst = assets_dir / f"{model_name}.pt"
    download_file(url, dst, expected_sha256=expected_sha256)
    return dst

