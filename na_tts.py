import asyncio
import shutil
import os
import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro
class TTTS:
    def __init__(
        self,
        voice: str = "af_heart",
        lang: str = "en-us",
        model_path: str = None,
        voices_path: str = None
    ):
        self.voice = voice
        self.lang = lang
        
        self.model_path = model_path
        self.voices_path = voices_path

        
        # Initialize Kokoro TTS engine
        self.tts = Kokoro(str(self.model_path), str(self.voices_path))
        self.shout_phrases = ["fuck","shit"]
    def is_shouting_phrase(self, text: str) -> bool:
        return any(phrase in text.lower() for phrase in self.shout_phrases)

    async def speak(self, text: str, speed: float = 1.0,pitch: float = 1.0):

            # Auto-adjust for shouting
        if self.is_shouting_phrase(text):
            print("Shouting detected! Applying shout effects.")
            gain_db = 25           # Add 5dB gain
            compress = True       # Enable compressor
        else:
            gain_db = 0
            compress = False


        
        all_samples = []
        sample_rate = None
        
        # Generate speech - CORRECTED METHOD NAME
        print("speeekşn")
        audio = self.tts.create_stream(
            text=text,
            voice=self.voice,
            lang=self.lang,
            speed=speed
        )
        
        async for samples, sr in audio:
            if sample_rate is None:
                sample_rate = sr
            all_samples.append(samples)
        #add spaces beetwen samples
        # Combine all audio chunks
        full_audio = np.concatenate(all_samples)
        if gain_db != 0:
            full_audio *= 10 ** (gain_db / 20)

        if compress:
            full_audio = self.compress_audio(full_audio)
        # Apply pitch shifting
        if pitch != 1.0:
            print("Ptichin")
            full_audio = self.pitch_shift(full_audio, sample_rate, pitch)
        
        # Play the modified audio
        sd.play(full_audio, sample_rate)
        sd.wait()

    def compress_audio(self, audio: np.ndarray) -> np.ndarray:
        # Simple dynamic range compression (soft knee)
        threshold = 0.2
        ratio = 4.0
        def compress_sample(sample):
            abs_sample = abs(sample)
            if abs_sample < threshold:
                return sample
            else:
                compressed = np.sign(sample) * (threshold + (abs_sample - threshold) / ratio)
                return compressed
        return np.array([compress_sample(s) for s in audio], dtype=np.float32)


    def pitch_shift(self, audio: np.ndarray, sample_rate: int, pitch_factor: float) -> np.ndarray:
        """Simple pitch shifting using resampling"""
        # Calculate new length based on pitch factor
        new_length = int(len(audio) / pitch_factor)
        
        # Generate linearly spaced indices for resampling
        indices = np.linspace(0, len(audio) - 1, new_length)
        
        # Interpolate to create pitch-shifted audio
        return np.interp(indices, np.arange(len(audio)), audio)

async def main():
    import torch
    use_gpu = torch.cuda.is_available()  # Returns True if GPU is detected
    print(use_gpu)
    import onnxruntime as ort
    print(ort.get_available_providers())
    # Specify the actual paths to your downloaded files
    MODEL_PATH = r"C:\Users\atoca\Desktop\Naka-chan\kokoro-v1.0.onnx"
    VOICES_PATH = r"C:\Users\atoca\Desktop\Naka-chan\voices-v1.0.bin"
    
    # Initialize TTS with explicit paths
    tts = TTTS(
        #"af_bella"
        voice="af_heart",
        model_path=MODEL_PATH,
        voices_path=VOICES_PATH
    )
    
    await tts.speak(
        "HAHAHAHAHAHAHAHAHAHA",
        speed=0.85,
        pitch= 1.2
    )

if __name__ == "__main__":
    asyncio.run(main())