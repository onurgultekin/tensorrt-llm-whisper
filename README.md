# TensorRT-LLM Whisper (C++ Runtime) on Modal

Bu repo, OpenAI Whisper modellerini **TensorRT-LLM C++ runtime** (Python binding: `ModelRunnerCpp`) ile çalıştırmak ve **Modal** üzerinde GPU’da servis etmek için bir iskelet sağlar.

## Ne var?

- `modal_app.py`: Modal App (engine build + inference + web endpoint)
- `trtllm_whisper/`: Whisper asset indir, checkpoint dönüştür, TRT engine build, inference
- Model dosyaları ve engine’ler Modal `Volume` içinde tutulur: `trtllm-whisper` → container içinde `/vol`

## Önkoşullar

- Modal hesabı + CLI kurulumu: `pip install modal` ve `modal setup`
- Modal üzerinde GPU (varsayılan: `A10`)

## 1) Engine build (ilk sefer)

Whisper checkpoint → TRT-LLM checkpoint → TensorRT engine(encoder/decoder).

```bash
modal run modal_app.py::build_whisper_engines --model-name large-v3
```

Bu işlem `/vol/assets` içine Whisper asset’lerini ve checkpoint’i indirir, `/vol/checkpoints` ve `/vol/engines` altına build çıktısını yazar ve volume’a `commit` eder.

## 2) Tek dosya transcribe (CLI)

```bash
modal run modal_app.py --audio-path /path/to/audio.wav --language tr
```

Parametreler:
- `--language`: Whisper dil token’ı (örn. `en`, `tr`, `de`…)
- `--task`: `transcribe` veya `translate`
- `--timestamps`: `true` ise timestamp token’larını kapatmaz
- `--timings`: container içi süre ölçümlerini de döndürür (`timings_ms`)

## 3) HTTP endpoint (FastAPI)

Dev için:

```bash
modal serve modal_app.py
```

Sonra multipart upload:

```bash
curl -F "file=@audio.wav" "https://<MODAL-URL>/transcribe?language=tr&task=transcribe"
```

Süre ölçümüyle:

```bash
curl -F "file=@audio.wav" "https://<MODAL-URL>/transcribe?language=tr&task=transcribe&timings=true"
```

## 4) WebSocket (Push-to-talk)

Amaç: tek bağlantı + 20ms chunk + client-side VAD ile “boş” ses göndermeden, **push-to-talk stop** anında transcribe etmek.

Server: `modal serve modal_app.py` ile `/ws` açılır.

Client (lokalde):

```bash
pip install sounddevice webrtcvad websockets
python ws_client.py --url wss://<MODAL-URL>/ws --language tr
```

- `Enter`: start/stop
- Varsayılan: sadece `end` sonrası `final` döner (connection açık kalır).
- İstersen otomatik segment için: `--segment-silence-ms 700` (pause sonrası `segment` mesajı)
- VAD sınırlarında kelime kırpılması olursa: `--pre-roll-ms 200` ve `--hangover-ms 400` değerlerini artırmayı veya `--vad 1`/`--vad 0` denemeyi düşün.

Prod için:

```bash
modal deploy modal_app.py
```

## 5) Load test (çoklu kullanıcı simülasyonu)

Basit bir concurrency testi için `ws_load_test.py` ile aynı anda birden fazla WebSocket client simüle edebilirsin.

```bash
pip install websockets
python ws_load_test.py --url wss://<MODAL-URL>/ws --users 5 --ramp-s 10 --sessions-per-user 2
```

Notlar:
- `whisper_sample.wav` (repo içinde) 16kHz mono PCM olduğu için varsayılan olarak onu kullanır.
- `--speed 1.0` gerçek zamanlıya yakın gönderir; `--speed 0` mümkün olduğunca hızlı gönderir.

## Notlar / Sık karşılaşılanlar

- Bu akış **Python runtime** değil; TensorRT-LLM’in **C++ runtime’ını** (`ModelRunnerCpp`) kullanır.
- `trtllm-build` tarafında `--paged_kv_cache enable` + `--remove_input_padding enable` ile inflight batching desteklenir.
- Varsayılan padding stratejisi `max` (30sn/3000 frame’e pad). Resmi OpenAI Whisper modelleri için en güvenlisi budur.
- İlk kurulumda image build + wheel indirme (özellikle `tensorrt_llm`) ve engine build uzun sürebilir.
- Aynı GPU üzerinde **Whisper + formatter LLM** birlikte yüklenecekse, TRT-LLM varsayılan olarak KV cache için GPU RAM’in büyük kısmını ayırmaya çalışır; gerekirse şu env’leri küçült:
  - `VOICEOS_WHISPER_KV_CACHE_FRACTION` (örn. `0.2`)
  - `VOICEOS_FORMATTER_KV_CACHE_FRACTION` (örn. `0.12`)
