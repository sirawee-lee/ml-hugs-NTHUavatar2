"""Speech I/O for the text2hugs pipeline.

STT: OpenAI Whisper (local)
TTS: HiggsAudio v2 by Boson AI (local)

Dependencies (install into the environment that runs run_text2hugs.py):
  pip install openai-whisper sounddevice scipy

  # HiggsAudio v2:
  git clone https://github.com/boson-ai/higgs-audio
  cd higgs-audio && pip install -e .
"""

import tempfile
from pathlib import Path

# Module-level cache so model weights are loaded only once per process.
_higgs_engine = None

# ── Browser recording page ────────────────────────────────────────────────────
_HTML_RECORD_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Whisper Input</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: sans-serif;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 100vh;
      background: #1a1a2e; color: #eee;
      gap: 1.2rem; padding: 2rem;
    }
    h1 { font-size: 1.6rem; }
    #btn {
      font-size: 1.4rem; padding: 0.9rem 2.2rem;
      border-radius: 2rem; border: none; cursor: pointer;
      background: #e94560; color: white;
      transition: background 0.2s;
    }
    #btn:disabled { background: #555; cursor: not-allowed; }
    #btn.recording { background: #c0392b; animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
    #status { font-size: 1.1rem; color: #a8dadc; text-align: center; }
    #transcript { font-size: 1rem; color: #f1c40f; max-width: 500px; text-align: center; }
  </style>
</head>
<body>
  <h1>&#127908; Speak your motion prompt</h1>
  <button id="btn">&#9654; Start Recording</button>
  <p id="status">Click the button and describe the motion you want to generate.</p>
  <p id="transcript"></p>
  <script>
    let mediaRecorder, chunks = [];
    const btn    = document.getElementById('btn');
    const status = document.getElementById('status');
    const trans  = document.getElementById('transcript');

    btn.onclick = async () => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        btn.disabled = true;
        btn.textContent = 'Uploading\u2026';
        btn.classList.remove('recording');
        return;
      }
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch(e) {
        status.textContent = '\u274c Mic access denied: ' + e.message;
        return;
      }
      mediaRecorder = new MediaRecorder(stream);
      chunks = [];
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        status.textContent = 'Uploading to server\u2026';
        try {
          const resp = await fetch('/upload', {
            method: 'POST',
            body: blob,
            headers: { 'Content-Type': blob.type }
          });
          if (resp.ok) {
            status.textContent = '\u2713 Audio received! Whisper is transcribing\u2026';
          } else {
            status.textContent = '\u274c Upload failed (HTTP ' + resp.status + ')';
            btn.disabled = false;
            btn.textContent = '\u25b6 Record Again';
          }
        } catch(e) {
          status.textContent = '\u274c Network error: ' + e.message;
          btn.disabled = false;
          btn.textContent = '\u25b6 Record Again';
        }
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
      btn.textContent = '\u23f9 Stop & Submit';
      btn.classList.add('recording');
      status.textContent = 'Recording\u2026 click Stop when done speaking.';
      trans.textContent = '';
    };
  </script>
</body>
</html>"""


# ── STT (Whisper) ─────────────────────────────────────────────────────────────

def record_via_browser(port: int = 9876) -> Path:
    """Start a local HTTP server, serve a recording page, and wait for audio upload.

    The user opens http://<lab-pc-ip>:<port> in their local browser, records
    from their local microphone, and clicks Stop. The audio is POSTed back and
    saved to a temp file which is returned.

    Works from any remote device (AnyDesk, SSH, VPN) because the browser
    captures from the client-side microphone — no audio forwarding needed.
    """
    import http.server
    import socket
    import threading

    received = threading.Event()
    result: dict = {}

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # suppress per-request noise

        def do_GET(self):
            if self.path == '/':
                body = _HTML_RECORD_PAGE.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == '/upload':
                length = int(self.headers.get('Content-Length', 0))
                audio_bytes = self.rfile.read(length)
                ct = self.headers.get('Content-Type', 'audio/webm')
                # Pick extension based on MIME type reported by the browser
                if 'ogg' in ct:
                    ext = '.ogg'
                elif 'wav' in ct:
                    ext = '.wav'
                else:
                    ext = '.webm'
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                result['path'] = Path(tmp.name)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'OK')
                received.set()
            else:
                self.send_response(404)
                self.end_headers()

    server = http.server.HTTPServer(('0.0.0.0', port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Show the URL the user should open on their local machine
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = '0.0.0.0'

    print(f"\n[STT] Open this URL in your LOCAL browser (on your own device):")
    print(f"[STT]   http://{local_ip}:{port}")
    print(f"[STT] Record your motion prompt and click Stop when done.")
    print(f"[STT] Waiting for audio upload...")

    received.wait()
    server.shutdown()
    print(f"[STT] Audio received from browser.")
    return result['path']


def record_audio(
    duration: float = 8.0,
    samplerate: int = 16000,
    alsa_device: str = "plughw:0,0",
) -> Path:
    """Record from the local microphone and save to a temporary WAV file.

    Tries sounddevice first (needs PortAudio); falls back to arecord with
    the specified ALSA device (bypasses PulseAudio, which on many Linux
    systems defaults to a monitor/loopback source rather than the mic).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = Path(tmp.name)
    tmp.close()

    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav

        print(f"\n[STT] Listening for {duration:.0f} seconds — speak now!")
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[STT] Recording finished.")
        wav.write(str(wav_path), samplerate, audio)

    except (ImportError, OSError):
        # PortAudio not available — use arecord with an explicit ALSA device.
        import subprocess
        print(f"\n[STT] Listening for {duration:.0f} seconds — speak now!"
              f" (arecord, device={alsa_device})")
        subprocess.run(
            [
                "arecord",
                "-D", alsa_device,
                "-d", str(int(duration)),
                "-r", str(samplerate),
                "-c", "1",
                "-f", "S16_LE",
                str(wav_path),
            ],
            check=True,
        )
        print("[STT] Recording finished.")

    return wav_path


def transcribe(audio_path: Path, model_size: str = "base") -> str:
    """Transcribe an audio file with Whisper and return the text.

    Accepts any format ffmpeg can decode (wav, webm, ogg, mp3, etc.).
    """
    import whisper

    print(f"[STT] Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)
    print(f"[STT] Transcribing {audio_path.name}...")
    result = model.transcribe(str(audio_path))
    text = result["text"].strip()
    print(f"[STT] Transcription: {text!r}")
    return text


def record_and_transcribe(
    duration: float = 8.0,
    samplerate: int = 16000,
    model_size: str = "base",
    alsa_device: str = "plughw:0,0",
) -> str:
    """Record from the local mic (ALSA) and return Whisper transcription."""
    wav_path = record_audio(duration=duration, samplerate=samplerate, alsa_device=alsa_device)
    try:
        return transcribe(wav_path, model_size=model_size)
    finally:
        wav_path.unlink(missing_ok=True)


def browser_record_and_transcribe(
    port: int = 9876,
    model_size: str = "base",
) -> str:
    """Capture audio from the user's remote browser and return Whisper transcription."""
    audio_path = record_via_browser(port=port)
    try:
        return transcribe(audio_path, model_size=model_size)
    finally:
        audio_path.unlink(missing_ok=True)


# ── TTS (HiggsAudio v2 with espeak fallback) ─────────────────────────────────

def _speak_espeak(text: str, out_wav: str | None = None) -> bool:
    """Speak text with espeak (lightweight fallback). Returns True on success."""
    import shutil
    import subprocess

    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        return False

    print(f"[TTS] espeak: {text!r}")
    if out_wav:
        subprocess.run([espeak, "-w", out_wav, text], check=True)
        print(f"[TTS] Audio saved to: {out_wav}")
    else:
        subprocess.run([espeak, text], check=True)
    return True


def speak_text(
    text: str,
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
    tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
    out_wav: str | None = None,
) -> None:
    """Synthesize speech and play/save it.

    Tries HiggsAudio v2 first; falls back to espeak/espeak-ng if HiggsAudio is
    not installed (espeak is always available on a standard Linux system).

    Args:
        text:           Text to speak.
        model_path:     HuggingFace model ID or local directory for HiggsAudio.
        tokenizer_path: HuggingFace tokenizer ID or local directory for HiggsAudio.
        out_wav:        If given, save the audio to this WAV path instead of playing.
    """
    global _higgs_engine

    try:
        import torch
        import torchaudio
        from boson_multimodal.data_types import ChatMLSample, Message
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    except ImportError as exc:
        print(f"[TTS] HiggsAudio v2 not available ({exc}). Trying espeak fallback...")
        if not _speak_espeak(text, out_wav=out_wav):
            print("[TTS] espeak not found either. No audio output.")
            print("[TTS] Install espeak:  sudo apt install espeak-ng")
            print("[TTS] Install HiggsAudio:  git clone https://github.com/boson-ai/higgs-audio && cd higgs-audio && pip install -e .")
        return

    if _higgs_engine is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Loading HiggsAudio v2 on {device} (first call only)...")
        _higgs_engine = HiggsAudioServeEngine(model_path, tokenizer_path, device=device)

    system_prompt = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
    )
    output = _higgs_engine.generate(
        chat_ml_sample=ChatMLSample(messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=text),
        ]),
        max_new_tokens=2048,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )

    audio_tensor = torch.from_numpy(output.audio)[None, :]
    print(f"[TTS] Speaking: {text!r}")

    if out_wav:
        torchaudio.save(out_wav, audio_tensor, output.sampling_rate)
        print(f"[TTS] Audio saved to: {out_wav}")
        return

    # Play back through speakers; fall back to a system player if sounddevice missing.
    try:
        import sounddevice as sd
        sd.play(output.audio, output.sampling_rate)
        sd.wait()
    except (ImportError, OSError):
        import os
        tmp_wav = tempfile.mktemp(suffix=".wav")
        torchaudio.save(tmp_wav, audio_tensor, output.sampling_rate)
        os.system(f"aplay '{tmp_wav}' 2>/dev/null || paplay '{tmp_wav}' 2>/dev/null")
        Path(tmp_wav).unlink(missing_ok=True)
