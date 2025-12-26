import argparse
import asyncio
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Optional

import sounddevice as sd
import webrtcvad
import websockets


SR = 16_000
FRAME_MS = 20  # WebRTC VAD supports 10/20/30 ms
SAMPLES_PER_FRAME = SR * FRAME_MS // 1000


@dataclass
class ClientState:
    recording: bool = False
    quitting: bool = False
    epoch: int = 0
    last_voice_send_ts: Optional[float] = None
    last_end_send_ts: Optional[float] = None


async def _stdin_loop(state: ClientState, toggle_q: asyncio.Queue[None]) -> None:
    print("Enter: start/stop | q+Enter: quit")
    while not state.quitting:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            state.quitting = True
            break
        if line.strip().lower() == "q":
            state.quitting = True
            break
        await toggle_q.put(None)


async def _recv_loop(ws, state: ClientState) -> None:
    async for msg in ws:
        try:
            data = json.loads(msg)
        except Exception:
            print("<<", msg)
            continue
        mtype = data.get("type")
        if mtype in ("segment", "final"):
            if mtype == "final" and state.last_end_send_ts is not None:
                data["client_latency_ms"] = (time.perf_counter() - state.last_end_send_ts) * 1000
            elif state.last_voice_send_ts is not None:
                data["client_latency_ms"] = (time.perf_counter() - state.last_voice_send_ts) * 1000
            print("<<", json.dumps(data, ensure_ascii=False))
        elif mtype in ("hello", "started", "reset_ok"):
            print("<<", data)
        elif mtype == "error":
            print("<< ERROR:", data.get("error"))
        else:
            print("<<", data)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="wss://.../ws")
    p.add_argument("--language", default="tr")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--timestamps", action="store_true", help="Enable timestamps tokens")
    p.add_argument("--vad", type=int, default=2, choices=[0, 1, 2, 3], help="WebRTC VAD aggressiveness")
    p.add_argument("--segment-silence-ms", type=int, default=0, help="Auto-segment after pause (0 disables)")
    p.add_argument("--pre-roll-ms", type=int, default=200, help="Send a little audio before first VAD speech frame")
    p.add_argument("--hangover-ms", type=int, default=400, help="Keep sending a bit after VAD stops detecting speech")
    args = p.parse_args()

    state = ClientState()
    toggle_q: asyncio.Queue[None] = asyncio.Queue()
    audio_q: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=200)

    vad = webrtcvad.Vad(args.vad)

    async with websockets.connect(args.url, max_size=None) as ws:
        recv_task = asyncio.create_task(_recv_loop(ws, state))
        stdin_task = asyncio.create_task(_stdin_loop(state, toggle_q))

        loop = asyncio.get_running_loop()

        def enqueue_chunk(epoch: int, chunk: bytes) -> None:
            if not state.recording:
                return
            if epoch != state.epoch:
                return
            try:
                audio_q.put_nowait((epoch, chunk))
            except asyncio.QueueFull:
                # Drop if client can't keep up to avoid growing latency.
                pass

        def audio_cb(indata, frames, time_info, status):
            if status:
                return
            if not state.recording:
                return
            chunk = bytes(indata)
            epoch = state.epoch
            loop.call_soon_threadsafe(enqueue_chunk, epoch, chunk)

        stream = sd.RawInputStream(
            samplerate=SR,
            channels=1,
            dtype="int16",
            blocksize=SAMPLES_PER_FRAME,
            callback=audio_cb,
        )

        async def audio_sender() -> None:
            pre_roll_frames = max(0, int(ceil(args.pre_roll_ms / FRAME_MS)))
            hangover_frames = max(0, int(ceil(args.hangover_ms / FRAME_MS)))
            pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
            in_speech = False
            hangover_left = 0
            current_epoch: Optional[int] = None

            while not state.quitting:
                epoch, chunk = await audio_q.get()
                if current_epoch != epoch:
                    current_epoch = epoch
                    in_speech = False
                    hangover_left = 0
                    pre_roll.clear()

                if (not state.recording) or (epoch != state.epoch):
                    continue
                if len(chunk) != SAMPLES_PER_FRAME * 2:
                    continue

                is_speech = vad.is_speech(chunk, SR)

                if in_speech:
                    if is_speech:
                        hangover_left = hangover_frames
                        await ws.send(chunk)
                        state.last_voice_send_ts = time.perf_counter()
                    else:
                        if hangover_left > 0:
                            hangover_left -= 1
                            await ws.send(chunk)
                            state.last_voice_send_ts = time.perf_counter()
                        else:
                            in_speech = False
                            pre_roll.clear()
                            if pre_roll_frames > 0:
                                pre_roll.append(chunk)
                else:
                    if is_speech:
                        if pre_roll_frames > 0:
                            for frame in pre_roll:
                                await ws.send(frame)
                                state.last_voice_send_ts = time.perf_counter()
                            pre_roll.clear()
                        in_speech = True
                        hangover_left = hangover_frames
                        await ws.send(chunk)
                        state.last_voice_send_ts = time.perf_counter()
                    else:
                        if pre_roll_frames > 0:
                            pre_roll.append(chunk)

        send_task = asyncio.create_task(audio_sender())

        with stream:
            while not state.quitting:
                await toggle_q.get()
                if state.quitting:
                    break

                if not state.recording:
                    state.epoch += 1
                    state.recording = True
                    state.last_voice_send_ts = None
                    state.last_end_send_ts = None
                    await ws.send(
                        json.dumps(
                            {
                                "type": "start",
                                "sr": SR,
                                "fmt": "pcm_s16le",
                                "language": args.language,
                                "task": args.task,
                                "timestamps": bool(args.timestamps),
                                "segment_silence_ms": args.segment_silence_ms,
                            }
                        )
                    )
                    print(">> started")
                else:
                    state.recording = False
                    state.last_end_send_ts = time.perf_counter()
                    await ws.send(json.dumps({"type": "end"}))
                    print(">> ended")

        state.quitting = True
        recv_task.cancel()
        send_task.cancel()
        stdin_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
