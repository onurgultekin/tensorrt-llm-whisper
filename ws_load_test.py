#!/usr/bin/env python3
"""
WebSocket load test for the `tensorrt-llm-whisper` Modal backend.

Simulates N concurrent push-to-talk clients streaming 16kHz mono PCM16LE to `/ws`.

Example:
  python ws_load_test.py --url wss://<MODAL-URL>/ws --users 5 --ramp-s 10 --sessions-per-user 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import websockets


SR = 16_000
SAMPLE_WIDTH_BYTES = 2


@dataclass
class ConnectResult:
    ws: Optional[websockets.WebSocketClientProtocol]
    connect_ms: Optional[float]
    connect_to_hello_ms: Optional[float]
    connect_attempts: int
    error: Optional[str]
    hello_ready: Optional[bool]
    hello_whisper_engine: Optional[bool]
    hello_formatter_engine: Optional[bool]
    hello_error_whisper: Optional[str]
    hello_error_formatter: Optional[str]


@dataclass(frozen=True)
class AudioProfile:
    name: str
    pcm: bytes

    @property
    def audio_sec(self) -> float:
        return _pcm_duration_sec(self.pcm)


@dataclass(frozen=True)
class SessionResult:
    user_id: int
    session_id: int
    audio_profile: str
    connect_ms: Optional[float]
    connect_attempts: int
    ok: bool
    error: Optional[str]
    audio_sec: float
    connect_to_hello_ms: Optional[float]
    hello_ready: Optional[bool]
    hello_whisper_engine: Optional[bool]
    hello_formatter_engine: Optional[bool]
    hello_error_whisper: Optional[str]
    hello_error_formatter: Optional[str]
    started_ms: Optional[float]
    first_segment_ms: Optional[float]
    start_to_final_ms: Optional[float]
    end_to_final_ms: Optional[float]
    whisper_server_total_ms: Optional[float]
    formatter_server_total_ms: Optional[float]
    formatter_llm_ms: Optional[float]
    formatter_prompt: Optional[str]
    formatter_pass: Optional[str]
    raw_text_preview: Optional[str]
    formatted_text_preview: Optional[str]
    segments_received: int
    log_lines: tuple[str, ...]


def _read_wav_pcm16le_mono_16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR:
            raise ValueError(f"{path} must be {SR} Hz (got {w.getframerate()})")
        if w.getnchannels() != 1:
            raise ValueError(f"{path} must be mono (got {w.getnchannels()} ch)")
        if w.getsampwidth() != SAMPLE_WIDTH_BYTES:
            raise ValueError(f"{path} must be 16-bit PCM (got sampwidth={w.getsampwidth()})")
        return w.readframes(w.getnframes())


def _pcm_duration_sec(pcm: bytes) -> float:
    return len(pcm) / float(SR * SAMPLE_WIDTH_BYTES)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty values")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _context_payload(mode: str, *, rng: random.Random) -> dict[str, Any]:
    m = mode.lower().strip()
    if m == "none":
        return {}
    if m == "unknown":
        return {"active_app": "Unknown app"}
    if m == "chatgpt":
        return {
            "active_app": "Web browser",
            "browser": {"domain": "chatgpt.com", "title": "Sayfa sayfa çeviri"},
        }
    if m == "gmail":
        return {
            "active_app": "Web browser",
            "browser": {"domain": "mail.google.com", "title": "Gmail"},
        }
    if m == "random":
        return _context_payload(rng.choice(["unknown", "chatgpt", "gmail"]), rng=rng)
    raise ValueError(f"Unknown context mode: {mode!r}")


async def _drain_until_type(
    ws: websockets.WebSocketClientProtocol,
    *,
    want_type: str,
    timeout_s: float,
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    while True:
        remaining = timeout_s - (time.perf_counter() - t0)
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {want_type!r}")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        t_recv = time.perf_counter()
        if isinstance(raw, (bytes, bytearray)):
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("type") == want_type:
            return msg, t_recv


def _preview(text: Optional[str], *, max_chars: int) -> Optional[str]:
    if text is None:
        return None
    t = text.strip()
    if max_chars <= 0:
        return t
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def _safe_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _choose_profile(
    *,
    session_id: int,
    rng: random.Random,
    scenario: str,
    mixed_mode: str,
    mixed_long_prob: float,
    short_profile: AudioProfile,
    long_profile: AudioProfile,
    single_profile: AudioProfile,
) -> AudioProfile:
    s = scenario.lower().strip()
    if s == "single":
        return single_profile
    if s == "short":
        return short_profile
    if s == "long":
        return long_profile
    if s != "mixed":
        raise ValueError(f"Unknown scenario: {scenario!r}")

    mm = mixed_mode.lower().strip()
    if mm == "alternate":
        return short_profile if (session_id % 2 == 0) else long_profile
    if mm == "random":
        return long_profile if (rng.random() < mixed_long_prob) else short_profile
    raise ValueError(f"Unknown mixed_mode: {mixed_mode!r}")


async def _run_user(
    *,
    user_id: int,
    url: str,
    frame_ms: int,
    speed: float,
    sessions_per_user: int,
    ramp_s: float,
    think_time_min_s: float,
    think_time_max_s: float,
    language: str,
    format_template: str,
    context_mode: str,
    rng_seed: int,
    connect_open_timeout_s: float,
    connect_retries: int,
    connect_backoff_initial_s: float,
    connect_backoff_max_s: float,
    connect_backoff_multiplier: float,
    timeout_final_s: float,
    hello_timeout_s: float,
    scenario: str,
    mixed_mode: str,
    mixed_long_prob: float,
    short_profile: AudioProfile,
    long_profile: AudioProfile,
    single_profile: AudioProfile,
    reuse_connection_per_user: bool,
    print_text_max_chars: int,
    log_segments_max: int,
) -> list[SessionResult]:
    rng = random.Random(rng_seed)

    frame_samples = int(SR * frame_ms / 1000)
    frame_bytes = frame_samples * SAMPLE_WIDTH_BYTES
    if frame_samples <= 0 or frame_bytes <= 0:
        raise ValueError(f"invalid frame_ms={frame_ms}")

    results: list[SessionResult] = []

    # Initial ramp delay so users start at different times.
    if ramp_s > 0:
        await asyncio.sleep(rng.random() * ramp_s)

    async def connect_once() -> ConnectResult:
        connect_attempts = 0
        last_connect_err: Optional[str] = None

        for attempt in range(max(1, connect_retries)):
            connect_attempts = attempt + 1
            ws: Optional[websockets.WebSocketClientProtocol] = None
            try:
                t_conn_start = time.perf_counter()
                ws = await websockets.connect(url, max_size=None, open_timeout=connect_open_timeout_s)
                t_conn_end = time.perf_counter()
                connect_ms = (t_conn_end - t_conn_start) * 1000

                hello_msg, t_hello = await _drain_until_type(ws, want_type="hello", timeout_s=hello_timeout_s)
                connect_to_hello_ms = (t_hello - t_conn_start) * 1000

                hello_ready = hello_msg.get("ready") if isinstance(hello_msg.get("ready"), bool) else None
                hello_whisper_engine = (
                    hello_msg.get("whisper_engine") if isinstance(hello_msg.get("whisper_engine"), bool) else None
                )
                hello_formatter_engine = (
                    hello_msg.get("formatter_engine") if isinstance(hello_msg.get("formatter_engine"), bool) else None
                )

                hello_error_whisper: Optional[str] = None
                hello_error_formatter: Optional[str] = None
                errors = hello_msg.get("errors")
                if isinstance(errors, dict):
                    ew = errors.get("whisper")
                    ef = errors.get("formatter")
                    if isinstance(ew, str):
                        hello_error_whisper = ew
                    elif ew is not None:
                        hello_error_whisper = repr(ew)
                    if isinstance(ef, str):
                        hello_error_formatter = ef
                    elif ef is not None:
                        hello_error_formatter = repr(ef)

                return ConnectResult(
                    ws=ws,
                    connect_ms=connect_ms,
                    connect_to_hello_ms=connect_to_hello_ms,
                    connect_attempts=connect_attempts,
                    error=None,
                    hello_ready=hello_ready,
                    hello_whisper_engine=hello_whisper_engine,
                    hello_formatter_engine=hello_formatter_engine,
                    hello_error_whisper=hello_error_whisper,
                    hello_error_formatter=hello_error_formatter,
                )
            except Exception as e:
                last_connect_err = repr(e)
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                if attempt + 1 >= max(1, connect_retries):
                    break
                base = connect_backoff_initial_s * (connect_backoff_multiplier**attempt)
                sleep_s = min(connect_backoff_max_s, max(0.0, base))
                sleep_s = sleep_s * (0.7 + 0.6 * rng.random())  # jitter
                await asyncio.sleep(sleep_s)

        return ConnectResult(
            ws=None,
            connect_ms=None,
            connect_to_hello_ms=None,
            connect_attempts=connect_attempts,
            error=last_connect_err,
            hello_ready=None,
            hello_whisper_engine=None,
            hello_formatter_engine=None,
            hello_error_whisper=None,
            hello_error_formatter=None,
        )

    def _ctx_summary(ctx: dict[str, Any]) -> str:
        active_app = ctx.get("active_app") if isinstance(ctx.get("active_app"), str) else None
        browser = ctx.get("browser") if isinstance(ctx.get("browser"), dict) else None
        domain = browser.get("domain") if isinstance(browser, dict) and isinstance(browser.get("domain"), str) else None
        title = browser.get("title") if isinstance(browser, dict) and isinstance(browser.get("title"), str) else None

        parts: list[str] = []
        if active_app:
            parts.append(f'app="{active_app}"')
        if domain:
            parts.append(f"domain={domain}")
        if title:
            parts.append(f'title="{title}"')
        return " ".join(parts) if parts else "(none)"

    async def run_session(
        *,
        conn: ConnectResult,
        session_id: int,
        include_connect_metrics: bool,
        profile: AudioProfile,
        ctx: dict[str, Any],
    ) -> SessionResult:
        if conn.ws is None:
            raise RuntimeError("run_session called without an active WebSocket")
        ws = conn.ws

        connect_ms = conn.connect_ms if include_connect_metrics else None
        connect_to_hello_ms = conn.connect_to_hello_ms if include_connect_metrics else None
        connect_attempts = conn.connect_attempts if include_connect_metrics else 1
        hello_ready = conn.hello_ready if include_connect_metrics else None
        hello_whisper_engine = conn.hello_whisper_engine if include_connect_metrics else None
        hello_formatter_engine = conn.hello_formatter_engine if include_connect_metrics else None
        hello_error_whisper = conn.hello_error_whisper if include_connect_metrics else None
        hello_error_formatter = conn.hello_error_formatter if include_connect_metrics else None

        events: list[tuple[str, float, dict[str, Any]]] = []

        final_msg: Optional[dict[str, Any]] = None
        final_t: Optional[float] = None

        async def recv_loop() -> None:
            nonlocal final_msg, final_t
            while True:
                raw = await ws.recv()
                t = time.perf_counter()
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if isinstance(mtype, str):
                    events.append((mtype, t, msg))
                if mtype == "final":
                    final_msg = msg
                    final_t = t
                    return

        recv_task = asyncio.create_task(recv_loop())

        t_start_sent = time.perf_counter()
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "sr": SR,
                    "fmt": "pcm_s16le",
                    "language": language,
                    "task": "transcribe",
                    "timestamps": False,
                    "format": True,
                    "format_template": format_template,
                    "format_context": ctx,
                    "format_language": language,
                }
            )
        )

        # Stream audio (realtime-ish if speed>0).
        sleep_s = (frame_ms / 1000.0) / speed if speed > 0 else 0.0
        pcm = profile.pcm
        for off in range(0, len(pcm), frame_bytes):
            chunk = pcm[off : off + frame_bytes]
            if not chunk:
                break
            if len(chunk) < frame_bytes:
                chunk = chunk + (b"\x00" * (frame_bytes - len(chunk)))
            await ws.send(chunk)
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

        t_end_sent = time.perf_counter()
        await ws.send(json.dumps({"type": "end"}))

        try:
            await asyncio.wait_for(recv_task, timeout=timeout_final_s)
        except TimeoutError:
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)
            raise TimeoutError(f"Timed out waiting for final (user={user_id} session={session_id})") from None

        if final_msg is None or final_t is None:
            raise RuntimeError("receiver exited without final message")

        # Derive a few client-visible timings.
        started_ms: Optional[float] = None
        first_segment_ms: Optional[float] = None
        segment_count = 0

        segment_events: list[dict[str, Any]] = []
        for mtype, t, _msg in events:
            if mtype == "started" and started_ms is None:
                started_ms = (t - t_start_sent) * 1000
            elif mtype == "segment":
                segment_count += 1
                if first_segment_ms is None:
                    first_segment_ms = (t - t_start_sent) * 1000
                seg_ms = (t - t_start_sent) * 1000
                seg_reason = _msg.get("reason") if isinstance(_msg.get("reason"), str) else None
                seg_audio_sec = _safe_float(_msg.get("audio_sec"))
                seg_timings = _msg.get("timings_ms") if isinstance(_msg.get("timings_ms"), dict) else None
                seg_server_total_ms = _safe_float(seg_timings.get("server_total") if isinstance(seg_timings, dict) else None)
                seg_text_preview = _preview(_msg.get("text") if isinstance(_msg.get("text"), str) else None, max_chars=print_text_max_chars)
                segment_events.append(
                    {
                        "ms": seg_ms,
                        "reason": seg_reason,
                        "audio_sec": seg_audio_sec,
                        "server_total_ms": seg_server_total_ms,
                        "text_preview": seg_text_preview,
                    }
                )

        end_to_final_ms = (final_t - t_end_sent) * 1000
        start_to_final_ms = (final_t - t_start_sent) * 1000

        last = final_msg.get("last") if isinstance(final_msg, dict) else None
        timings_ms = last.get("timings_ms") if isinstance(last, dict) else None

        whisper_server_total_ms = None
        if isinstance(timings_ms, dict):
            whisper_server_total_ms = _safe_float(timings_ms.get("server_total"))

        fmt = final_msg.get("format") if isinstance(final_msg, dict) else None
        formatter_server_total_ms = None
        formatter_llm_ms = None
        formatter_prompt = None
        formatter_pass = None
        raw_text_preview = None
        formatted_text_preview = None

        if isinstance(fmt, dict):
            formatter_server_total_ms = _safe_float(fmt.get("server_total_ms"))
            formatter_llm_ms = _safe_float(fmt.get("llm_ms"))
            pv = fmt.get("prompt_version")
            if isinstance(pv, str):
                formatter_prompt = pv
            up = fmt.get("used_pass")
            if isinstance(up, str):
                formatter_pass = up

            raw_text_preview = _preview(fmt.get("raw_text") if isinstance(fmt.get("raw_text"), str) else None, max_chars=print_text_max_chars)
            formatted_text_preview = _preview(
                fmt.get("selected_text") if isinstance(fmt.get("selected_text"), str) else None,
                max_chars=print_text_max_chars,
            )
        else:
            raw_text_preview = _preview(final_msg.get("text") if isinstance(final_msg.get("text"), str) else None, max_chars=print_text_max_chars)
            formatted_text_preview = _preview(
                final_msg.get("formatted_text") if isinstance(final_msg.get("formatted_text"), str) else None,
                max_chars=print_text_max_chars,
            )

        # Build per-session log lines (printed later to avoid interleaving).
        log: list[str] = []
        log.append(f"--- user={user_id} session={session_id} profile={profile.name} audio={profile.audio_sec:.2f}s ---")
        log.append(f"context: {_ctx_summary(ctx)}")
        if connect_ms is not None:
            log.append(f"connect: {connect_ms:.0f}ms (attempts={connect_attempts})")
            if connect_to_hello_ms is not None:
                log.append(
                    "hello: "
                    f"{connect_to_hello_ms:.0f}ms "
                    f"(ready={hello_ready} whisper_engine={hello_whisper_engine} formatter_engine={hello_formatter_engine})"
                )
                if hello_error_whisper:
                    log.append(f"hello.errors.whisper: {_preview(hello_error_whisper, max_chars=160)}")
                if hello_error_formatter:
                    log.append(f"hello.errors.formatter: {_preview(hello_error_formatter, max_chars=160)}")
        else:
            log.append(f"connect: reused (attempts={connect_attempts})")
        log.append(f"start→started: {started_ms:.0f}ms" if started_ms is not None else "start→started: (missing)")
        if segment_count:
            log.append(f"start→first_segment: {first_segment_ms:.0f}ms (segments={segment_count})" if first_segment_ms is not None else f"segments: {segment_count}")
        else:
            log.append("segments: 0")
        log.append(f"end→final: {end_to_final_ms:.0f}ms  start→final: {start_to_final_ms:.0f}ms")
        if whisper_server_total_ms is not None:
            log.append(f"whisper.server_total: {whisper_server_total_ms:.0f}ms")
        if segment_events:
            show = segment_events if log_segments_max < 0 else segment_events[: max(0, log_segments_max)]
            for i, seg in enumerate(show, start=1):
                ms = seg.get("ms")
                reason = seg.get("reason") or "-"
                asec = seg.get("audio_sec")
                st = seg.get("server_total_ms")
                txt = seg.get("text_preview")
                bits = [f"segment#{i} @ {ms:.0f}ms", f"reason={reason}"]
                if isinstance(asec, (int, float)):
                    bits.append(f"audio={float(asec):.2f}s")
                if isinstance(st, (int, float)):
                    bits.append(f"server_total={float(st):.0f}ms")
                if isinstance(txt, str) and txt:
                    bits.append(f'text="{txt}"')
                log.append("  " + " ".join(bits))
            if log_segments_max >= 0 and len(segment_events) > log_segments_max:
                log.append(f"  … {len(segment_events) - log_segments_max} more segments omitted")
        if formatter_llm_ms is not None:
            log.append(f"formatter.llm_ms: {formatter_llm_ms:.0f}ms  used_pass={formatter_pass or '-'}  prompt={formatter_prompt or '-'}")
        if raw_text_preview is not None:
            log.append(f"raw: {raw_text_preview}")
        if formatted_text_preview is not None:
            log.append(f"formatted: {formatted_text_preview}")

        # Include safety outcomes per pass if present (short summary).
        if isinstance(fmt, dict):
            passes = fmt.get("passes")
            if isinstance(passes, list) and passes:
                for pinfo in passes:
                    if not isinstance(pinfo, dict):
                        continue
                    pname = pinfo.get("pass")
                    safety = pinfo.get("safety")
                    if not isinstance(pname, str) or not isinstance(safety, dict):
                        continue
                    ok_flag = safety.get("ok")
                    new_cnt = safety.get("out_new_token_count")
                    subseq = safety.get("subsequence")
                    sample = safety.get("out_new_tokens_sample")
                    sample_str = ""
                    if isinstance(sample, list) and sample:
                        sample_str = f" sample={sample[:6]}"
                    log.append(
                        f"pass={pname} ok={ok_flag} new_tokens={new_cnt} subseq={subseq}{sample_str}"
                    )

        return SessionResult(
            user_id=user_id,
            session_id=session_id,
            audio_profile=profile.name,
            connect_ms=connect_ms,
            connect_attempts=connect_attempts,
            ok=True,
            error=None,
            audio_sec=profile.audio_sec,
            connect_to_hello_ms=connect_to_hello_ms,
            hello_ready=hello_ready,
            hello_whisper_engine=hello_whisper_engine,
            hello_formatter_engine=hello_formatter_engine,
            hello_error_whisper=hello_error_whisper,
            hello_error_formatter=hello_error_formatter,
            started_ms=started_ms,
            first_segment_ms=first_segment_ms,
            start_to_final_ms=start_to_final_ms,
            end_to_final_ms=end_to_final_ms,
            whisper_server_total_ms=whisper_server_total_ms,
            formatter_server_total_ms=formatter_server_total_ms,
            formatter_llm_ms=formatter_llm_ms,
            formatter_prompt=formatter_prompt,
            formatter_pass=formatter_pass,
            raw_text_preview=raw_text_preview,
            formatted_text_preview=formatted_text_preview,
            segments_received=segment_count,
            log_lines=tuple(log),
        )

    if reuse_connection_per_user:
        conn = await connect_once()
        if conn.ws is None:
            # One failed connect aborts all sessions for this user (matches real client behavior).
            for session_id in range(sessions_per_user):
                prof = _choose_profile(
                    session_id=session_id,
                    rng=rng,
                    scenario=scenario,
                    mixed_mode=mixed_mode,
                    mixed_long_prob=mixed_long_prob,
                    short_profile=short_profile,
                    long_profile=long_profile,
                    single_profile=single_profile,
                )
                results.append(
                    SessionResult(
                        user_id=user_id,
                        session_id=session_id,
                        audio_profile=prof.name,
                        connect_ms=conn.connect_ms,
                        connect_attempts=conn.connect_attempts,
                        ok=False,
                        error=f"connect_failed: {conn.error}",
                        audio_sec=prof.audio_sec,
                        connect_to_hello_ms=conn.connect_to_hello_ms,
                        hello_ready=conn.hello_ready,
                        hello_whisper_engine=conn.hello_whisper_engine,
                        hello_formatter_engine=conn.hello_formatter_engine,
                        hello_error_whisper=conn.hello_error_whisper,
                        hello_error_formatter=conn.hello_error_formatter,
                        started_ms=None,
                        first_segment_ms=None,
                        start_to_final_ms=None,
                        end_to_final_ms=None,
                        whisper_server_total_ms=None,
                        formatter_server_total_ms=None,
                        formatter_llm_ms=None,
                        formatter_prompt=None,
                        formatter_pass=None,
                        raw_text_preview=None,
                        formatted_text_preview=None,
                        segments_received=0,
                        log_lines=(f"--- user={user_id} connect failed: {conn.error} ---",),
                    )
                )
            return results

        try:
            for session_id in range(sessions_per_user):
                if session_id > 0 and think_time_max_s > 0:
                    think = rng.uniform(max(0.0, think_time_min_s), max(think_time_min_s, think_time_max_s))
                    await asyncio.sleep(think)

                prof = _choose_profile(
                    session_id=session_id,
                    rng=rng,
                    scenario=scenario,
                    mixed_mode=mixed_mode,
                    mixed_long_prob=mixed_long_prob,
                    short_profile=short_profile,
                    long_profile=long_profile,
                    single_profile=single_profile,
                )
                ctx = _context_payload(context_mode, rng=rng)

                try:
                    res = await run_session(
                        conn=conn,
                        session_id=session_id,
                        include_connect_metrics=(session_id == 0),
                        profile=prof,
                        ctx=ctx,
                    )
                    results.append(res)
                except Exception as e:
                    include_connect = session_id == 0
                    results.append(
                        SessionResult(
                            user_id=user_id,
                            session_id=session_id,
                            audio_profile=prof.name,
                            connect_ms=conn.connect_ms if include_connect else None,
                            connect_attempts=conn.connect_attempts if include_connect else 1,
                            ok=False,
                            error=repr(e),
                            audio_sec=prof.audio_sec,
                            connect_to_hello_ms=conn.connect_to_hello_ms if include_connect else None,
                            hello_ready=conn.hello_ready if include_connect else None,
                            hello_whisper_engine=conn.hello_whisper_engine if include_connect else None,
                            hello_formatter_engine=conn.hello_formatter_engine if include_connect else None,
                            hello_error_whisper=conn.hello_error_whisper if include_connect else None,
                            hello_error_formatter=conn.hello_error_formatter if include_connect else None,
                            started_ms=None,
                            first_segment_ms=None,
                            start_to_final_ms=None,
                            end_to_final_ms=None,
                            whisper_server_total_ms=None,
                            formatter_server_total_ms=None,
                            formatter_llm_ms=None,
                            formatter_prompt=None,
                            formatter_pass=None,
                            raw_text_preview=None,
                            formatted_text_preview=None,
                            segments_received=0,
                            log_lines=(f"--- user={user_id} session={session_id} failed: {e!r} ---",),
                        )
                    )
        finally:
            try:
                await conn.ws.close()
            except Exception:
                pass
        return results

    # Legacy mode: new WS connection per session.
    for session_id in range(sessions_per_user):
        if session_id > 0 and think_time_max_s > 0:
            think = rng.uniform(max(0.0, think_time_min_s), max(think_time_min_s, think_time_max_s))
            await asyncio.sleep(think)

        prof = _choose_profile(
            session_id=session_id,
            rng=rng,
            scenario=scenario,
            mixed_mode=mixed_mode,
            mixed_long_prob=mixed_long_prob,
            short_profile=short_profile,
            long_profile=long_profile,
            single_profile=single_profile,
        )
        ctx = _context_payload(context_mode, rng=rng)

        conn = await connect_once()
        if conn.ws is None:
            results.append(
                SessionResult(
                    user_id=user_id,
                    session_id=session_id,
                    audio_profile=prof.name,
                    connect_ms=conn.connect_ms,
                    connect_attempts=conn.connect_attempts,
                    ok=False,
                    error=f"connect_failed: {conn.error}",
                    audio_sec=prof.audio_sec,
                    connect_to_hello_ms=conn.connect_to_hello_ms,
                    hello_ready=conn.hello_ready,
                    hello_whisper_engine=conn.hello_whisper_engine,
                    hello_formatter_engine=conn.hello_formatter_engine,
                    hello_error_whisper=conn.hello_error_whisper,
                    hello_error_formatter=conn.hello_error_formatter,
                    started_ms=None,
                    first_segment_ms=None,
                    start_to_final_ms=None,
                    end_to_final_ms=None,
                    whisper_server_total_ms=None,
                    formatter_server_total_ms=None,
                    formatter_llm_ms=None,
                    formatter_prompt=None,
                    formatter_pass=None,
                    raw_text_preview=None,
                    formatted_text_preview=None,
                    segments_received=0,
                    log_lines=(f"--- user={user_id} session={session_id} connect failed: {conn.error} ---",),
                )
            )
            continue

        try:
            res = await run_session(conn=conn, session_id=session_id, include_connect_metrics=True, profile=prof, ctx=ctx)
            results.append(res)
        except Exception as e:
            results.append(
                SessionResult(
                    user_id=user_id,
                    session_id=session_id,
                    audio_profile=prof.name,
                    connect_ms=conn.connect_ms,
                    connect_attempts=conn.connect_attempts,
                    ok=False,
                    error=repr(e),
                    audio_sec=prof.audio_sec,
                    connect_to_hello_ms=conn.connect_to_hello_ms,
                    hello_ready=conn.hello_ready,
                    hello_whisper_engine=conn.hello_whisper_engine,
                    hello_formatter_engine=conn.hello_formatter_engine,
                    hello_error_whisper=conn.hello_error_whisper,
                    hello_error_formatter=conn.hello_error_formatter,
                    started_ms=None,
                    first_segment_ms=None,
                    start_to_final_ms=None,
                    end_to_final_ms=None,
                    whisper_server_total_ms=None,
                    formatter_server_total_ms=None,
                    formatter_llm_ms=None,
                    formatter_prompt=None,
                    formatter_pass=None,
                    raw_text_preview=None,
                    formatted_text_preview=None,
                    segments_received=0,
                    log_lines=(f"--- user={user_id} session={session_id} failed: {e!r} ---",),
                )
            )
        finally:
            try:
                await conn.ws.close()
            except Exception:
                pass

    return results


def _print_summary(results: list[SessionResult]) -> None:
    total = len(results)
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print("\n=== Summary ===")
    print(f"sessions_total: {total}")
    print(f"sessions_ok:    {len(ok)}")
    print(f"sessions_fail:  {len(failed)}")

    if failed:
        by_err: dict[str, int] = {}
        for r in failed:
            by_err[r.error or "unknown"] = by_err.get(r.error or "unknown", 0) + 1
        top = sorted(by_err.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("\nTop errors:")
        for err, n in top:
            print(f"- {n}× {err}")

    def collect(getter):
        xs: list[float] = []
        for r in ok:
            v = getter(r)
            if isinstance(v, (int, float)):
                xs.append(float(v))
        return xs

    connect_ms = collect(lambda r: r.connect_ms)
    connect_to_hello_ms = collect(lambda r: r.connect_to_hello_ms)
    start_to_started_ms = collect(lambda r: r.started_ms)
    start_to_first_segment_ms = collect(lambda r: r.first_segment_ms)
    start_to_final_ms = collect(lambda r: r.start_to_final_ms)
    e2e = collect(lambda r: r.end_to_final_ms)
    whisper = collect(lambda r: r.whisper_server_total_ms)
    formatter_server_total = collect(lambda r: r.formatter_server_total_ms)
    llm = collect(lambda r: r.formatter_llm_ms)

    if connect_ms:
        print("\nconnect_ms:")
        print(
            f"  p50={_percentile(connect_ms, 50):.0f}  p90={_percentile(connect_ms, 90):.0f}  p99={_percentile(connect_ms, 99):.0f}  mean={statistics.mean(connect_ms):.0f}"
        )
    if connect_to_hello_ms:
        print("\nconnect_to_hello_ms:")
        print(
            f"  p50={_percentile(connect_to_hello_ms, 50):.0f}  p90={_percentile(connect_to_hello_ms, 90):.0f}  p99={_percentile(connect_to_hello_ms, 99):.0f}  mean={statistics.mean(connect_to_hello_ms):.0f}"
        )
    if start_to_started_ms:
        print("\nstart_to_started_ms:")
        print(
            f"  p50={_percentile(start_to_started_ms, 50):.0f}  p90={_percentile(start_to_started_ms, 90):.0f}  p99={_percentile(start_to_started_ms, 99):.0f}  mean={statistics.mean(start_to_started_ms):.0f}"
        )
    if start_to_first_segment_ms:
        print("\nstart_to_first_segment_ms:")
        print(
            f"  p50={_percentile(start_to_first_segment_ms, 50):.0f}  p90={_percentile(start_to_first_segment_ms, 90):.0f}  p99={_percentile(start_to_first_segment_ms, 99):.0f}  mean={statistics.mean(start_to_first_segment_ms):.0f}"
        )
    if start_to_final_ms:
        print("\nstart_to_final_ms:")
        print(
            f"  p50={_percentile(start_to_final_ms, 50):.0f}  p90={_percentile(start_to_final_ms, 90):.0f}  p99={_percentile(start_to_final_ms, 99):.0f}  mean={statistics.mean(start_to_final_ms):.0f}"
        )
    if e2e:
        print("\nend_to_final_ms:")
        print(f"  p50={_percentile(e2e, 50):.0f}  p90={_percentile(e2e, 90):.0f}  p99={_percentile(e2e, 99):.0f}  mean={statistics.mean(e2e):.0f}")
    if whisper:
        print("\nwhisper_server_total_ms:")
        print(f"  p50={_percentile(whisper, 50):.0f}  p90={_percentile(whisper, 90):.0f}  p99={_percentile(whisper, 99):.0f}  mean={statistics.mean(whisper):.0f}")
    if formatter_server_total:
        print("\nformatter_server_total_ms:")
        print(
            f"  p50={_percentile(formatter_server_total, 50):.0f}  p90={_percentile(formatter_server_total, 90):.0f}  p99={_percentile(formatter_server_total, 99):.0f}  mean={statistics.mean(formatter_server_total):.0f}"
        )
    if llm:
        print("\nformatter_llm_ms:")
        print(f"  p50={_percentile(llm, 50):.0f}  p90={_percentile(llm, 90):.0f}  p99={_percentile(llm, 99):.0f}  mean={statistics.mean(llm):.0f}")

    prompt_versions = sorted({r.formatter_prompt for r in ok if r.formatter_prompt})
    if prompt_versions:
        print("\nprompt_versions:", ", ".join(prompt_versions))
    used_passes = sorted({r.formatter_pass for r in ok if r.formatter_pass})
    if used_passes:
        print("used_passes:", ", ".join(used_passes))


def _print_per_session_logs(results: list[SessionResult]) -> None:
    print("\n=== Per-session details ===")
    for r in sorted(results, key=lambda x: (x.user_id, x.session_id)):
        for line in r.log_lines:
            print(line)


def _print_per_user_logs(results: list[SessionResult]) -> None:
    print("\n=== Per-user details ===")
    by_user: dict[int, list[SessionResult]] = {}
    for r in results:
        by_user.setdefault(r.user_id, []).append(r)
    for user_id in sorted(by_user.keys()):
        print(f"\n##### user={user_id} #####")
        for r in sorted(by_user[user_id], key=lambda x: x.session_id):
            for line in r.log_lines:
                print(line)


async def _amain() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="wss://.../ws")
    p.add_argument("--users", type=int, default=5)
    p.add_argument("--sessions-per-user", type=int, default=1)
    p.add_argument("--ramp-s", type=float, default=10.0, help="Random initial delay per user in [0,ramp-s]")
    p.add_argument("--think-time-min-s", type=float, default=0.0)
    p.add_argument("--think-time-max-s", type=float, default=0.0)
    p.add_argument("--audio-wav", type=str, default=str(Path(__file__).with_name("whisper_sample.wav")))
    p.add_argument("--audio-repeat", type=int, default=1, help="Repeat the WAV audio N times to make it longer")
    p.add_argument("--scenario", type=str, default="single", choices=["single", "short", "long", "mixed"])
    p.add_argument("--short-sec", type=float, default=2.0, help="For --scenario short/mixed: trim audio to first N seconds")
    p.add_argument("--long-repeat", type=int, default=16, help="For --scenario long/mixed: repeat the WAV audio N times")
    p.add_argument("--mixed-mode", type=str, default="alternate", choices=["alternate", "random"])
    p.add_argument("--mixed-long-prob", type=float, default=0.5, help="For --mixed-mode random: probability of long session")
    p.add_argument("--frame-ms", type=int, default=20, help="PCM frame size to send")
    p.add_argument("--speed", type=float, default=1.0, help="1.0=real-time, 2.0=2x, 0=send as fast as possible")
    p.add_argument("--language", type=str, default="en")
    p.add_argument("--format-template", type=str, default="auto", choices=["auto", "generic", "chat_prompt", "email", "doc"])
    p.add_argument("--context", type=str, default="random", choices=["random", "none", "unknown", "chatgpt", "gmail"])
    p.add_argument("--reuse-connection-per-user", action="store_true", help="Reuse one WS connection per user across sessions")
    p.add_argument("--print-text-max-chars", type=int, default=200, help="0 to print full text previews")
    p.add_argument("--log-mode", type=str, default="user", choices=["user", "session"], help="How to print detailed logs")
    p.add_argument("--no-per-session-log", action="store_true", help="Only print summary (kept for backwards compatibility)")
    p.add_argument("--connect-open-timeout-s", type=float, default=60.0, help="WebSocket opening handshake timeout")
    p.add_argument("--hello-timeout-s", type=float, default=10.0, help="Timeout waiting for initial hello after connect")
    p.add_argument("--connect-retries", type=int, default=10, help="Connection attempts per session")
    p.add_argument("--connect-backoff-initial-s", type=float, default=0.5)
    p.add_argument("--connect-backoff-max-s", type=float, default=8.0)
    p.add_argument("--connect-backoff-multiplier", type=float, default=1.6)
    p.add_argument("--timeout-final-s", type=float, default=60.0)
    p.add_argument("--log-segments-max", type=int, default=3, help="Max segment events to print per session (-1 for all)")
    args = p.parse_args()

    audio_path = Path(args.audio_wav).expanduser()
    base_pcm = _read_wav_pcm16le_mono_16k(audio_path)
    single_pcm = base_pcm * max(1, args.audio_repeat)

    short_frames = int(max(0.1, args.short_sec) * SR)
    short_pcm = base_pcm[: short_frames * SAMPLE_WIDTH_BYTES]
    if not short_pcm:
        short_pcm = base_pcm

    long_pcm = base_pcm * max(1, args.long_repeat)

    short_profile = AudioProfile(name="short", pcm=short_pcm)
    long_profile = AudioProfile(name="long", pcm=long_pcm)
    single_profile = AudioProfile(name="single", pcm=single_pcm)

    tasks = []
    base_seed = int(time.time()) & 0xFFFF_FFFF
    for user_id in range(args.users):
        tasks.append(
            asyncio.create_task(
                _run_user(
                    user_id=user_id,
                    url=args.url,
                    frame_ms=args.frame_ms,
                    speed=args.speed,
                    sessions_per_user=args.sessions_per_user,
                    ramp_s=max(0.0, args.ramp_s),
                    think_time_min_s=max(0.0, args.think_time_min_s),
                    think_time_max_s=max(0.0, args.think_time_max_s),
                    language=args.language,
                    format_template=args.format_template,
                    context_mode=args.context,
                    rng_seed=base_seed + user_id,
                    connect_open_timeout_s=max(1.0, args.connect_open_timeout_s),
                    connect_retries=max(1, args.connect_retries),
                    connect_backoff_initial_s=max(0.0, args.connect_backoff_initial_s),
                    connect_backoff_max_s=max(0.0, args.connect_backoff_max_s),
                    connect_backoff_multiplier=max(1.0, args.connect_backoff_multiplier),
                    timeout_final_s=max(1.0, args.timeout_final_s),
                    hello_timeout_s=max(0.1, args.hello_timeout_s),
                    scenario=args.scenario,
                    mixed_mode=args.mixed_mode,
                    mixed_long_prob=min(1.0, max(0.0, args.mixed_long_prob)),
                    short_profile=short_profile,
                    long_profile=long_profile,
                    single_profile=single_profile,
                    reuse_connection_per_user=bool(args.reuse_connection_per_user),
                    print_text_max_chars=int(args.print_text_max_chars),
                    log_segments_max=int(args.log_segments_max),
                )
            )
        )

    all_results: list[SessionResult] = []
    for res in await asyncio.gather(*tasks):
        all_results.extend(res)

    if not args.no_per_session_log:
        if args.log_mode == "session":
            _print_per_session_logs(all_results)
        else:
            _print_per_user_logs(all_results)
    _print_summary(all_results)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
