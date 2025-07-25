import asyncio
import sounddevice as sd
import numpy as np
import threading
from collections import deque
import noisereduce as nr
from faster_whisper import WhisperModel

class EnergyVADWhisper:
    def __init__(
        self,
        model_size="distil-medium.en",
        device=None,
        sample_rate=16_000,
        block_duration=0.10,
        energy_threshold=0.09,
        silence_duration=0.9
    ):
        import torch
        if device is None:
            device = "cuda"
        self.model = WhisperModel(model_size, device=device, compute_type="float32")
        self.sample_rate = sample_rate

        self.block_size = int(self.sample_rate * block_duration)
        self.energy_threshold = energy_threshold
        self.silence_frames = int(silence_duration / block_duration)

        self.pre_speech = deque(maxlen=self.silence_frames)
        self.speech_buffer = []

        self.in_speech = False
        self.silence_counter = 0

        # --- new: an asyncio.Queue for passing transcripts back to the async world ---
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _callback(self, indata, frames, time, status):
        audio_block = indata[:,0].copy()
        rms = np.sqrt(np.mean(audio_block**2))

        if self.in_speech:
            self.speech_buffer.append(audio_block)
            if rms < self.energy_threshold:
                self.silence_counter += 1
                if self.silence_counter >= self.silence_frames:
                    self._finalize_utterance()
            else:
                self.silence_counter = 0
        else:
            self.pre_speech.append(audio_block)
            if rms >= self.energy_threshold:
                self.in_speech = True
                self.silence_counter = 0
                self.speech_buffer = list(self.pre_speech)
                self.pre_speech.clear()

    def _finalize_utterance(self):
        self.in_speech = False
        self.silence_counter = 0

        audio = np.concatenate(self.speech_buffer)
        self.speech_buffer = []

        # spawn transcription in a real thread (so we don't block the audio callback)
        threading.Thread(target=self._transcribe, args=(audio,)).start()

    def _transcribe(self, audio_array):
        peak = np.max(np.abs(audio_array))
        if peak > 0:
            audio_array = audio_array / peak * 0.9  # Scale to 90% peak
        # do the Whisper work
        cleaned = nr.reduce_noise(y=audio_array, sr=self.sample_rate)
        segments, _ = self.model.transcribe(cleaned, language="en", beam_size=4)
        text = " ".join(seg.text for seg in segments).strip()
        print("🗣️ Recognized:", text)

        # push the result onto the asyncio queue
        if self._loop is not None:
            # thread‐safe enqueue
            self._loop.call_soon_threadsafe(self._queue.put_nowait, text)

    async def listen_realtime(self):
        self._loop = asyncio.get_running_loop()

        # 1) flush any leftover transcripts
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        print(f"🎧 Listening (energy_threshold={self.energy_threshold})")
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self._callback
        ):
            # 2) now wait for exactly one new utterance
            text = await self._queue.get()
            return text
