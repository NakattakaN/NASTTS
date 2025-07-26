# na_stt.py
import asyncio
import threading
from collections import deque

import numpy as np
import noisereduce as nr
import sounddevice as sd
from faster_whisper import WhisperModel


class FastEnergyVADWhisper:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str | None = None,
        sample_rate: int = 16_000,
        block_duration: float = 0.05,
        hop_duration: float   = 0.025,       # shorter blocks for lower latency
        energy_threshold: float = 0.04,
        silence_duration: float = 0.9       # shorter silence window
    ):
        # choose CUDA if available
        if device is None:
            import torch
            device = "cuda"

        # INT8 via CTranslate2 for speed
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8_float16",
            download_root=None
        )

        self.sample_rate = sample_rate
        # number of samples per block window
        self.block_size = int(self.sample_rate * block_duration)
        # number of samples per hop stride
        self.hop_size   = int(self.sample_rate * hop_duration)
        # VAD thresholds
        self.energy_threshold = energy_threshold
        # how many hops of silence to end utterance
        self.silence_hops = int(silence_duration / hop_duration)

        # rolling buffer for full block context
        self.window_buffer = deque(maxlen=self.block_size)
        # pre-speech buffer in hops
        self.pre_speech = deque(maxlen=self.silence_hops)
        # collected speech hops
        self.speech_buffer: list[np.ndarray] = []

        self.in_speech = False
        self.silence_counter = 0

        # asyncio queue for results
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _callback(self, indata, frames, time, status):
        # each callback provides hop_size samples
        audio_hop = indata[:, 0].copy()
        # maintain rolling window (unused directly but available if needed)
        self.window_buffer.extend(audio_hop)

        # compute RMS on this hop
        rms = np.sqrt(np.mean(audio_hop**2))

        if self.in_speech:
            self.speech_buffer.append(audio_hop)
            if rms < self.energy_threshold:
                self.silence_counter += 1
                if self.silence_counter >= self.silence_hops:
                    self._finalize_utterance()
            else:
                self.silence_counter = 0
        else:
            # waiting for speech start
            self.pre_speech.append(audio_hop)
            if rms >= self.energy_threshold:
                # speech start detected, include pre-speech hops
                self.in_speech = True
                self.silence_counter = 0
                # initialize buffer with preceding hops
                self.speech_buffer = list(self.pre_speech)
                self.pre_speech.clear()

    def _finalize_utterance(self):
        self.in_speech = False
        self.silence_counter = 0

        # concatenate all hops into one array
        audio = np.concatenate(self.speech_buffer)
        self.speech_buffer = []

        # offload transcription
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray):
        # normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9

        # noise reduction
        cleaned = nr.reduce_noise(y=audio, sr=self.sample_rate)

        # fast whisper transcription
        segments, _ = self.model.transcribe(
            cleaned,
            language="en",
            beam_size=1,                  # beam_size=1 for speed
            vad_filter=True,              # built‑in VAD post‑filter
        )
        text = " ".join(s.text for s in segments).strip()
        # send back to asyncio loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, text)

    async def listen_realtime(self) -> str:
        self._loop = asyncio.get_running_loop()
        # clear any stale items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        print(f"🎧 Fast Listening (thr={self.energy_threshold})")
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.hop_size,     # fire callback every hop
            callback=self._callback
        ):
            # await next utterance
            return await self._queue.get()
