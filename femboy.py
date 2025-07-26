from na_tts import TTTS
import keyboard
from na_stt import FastEnergyVADWhisper as EnergyVADWhisper
import asyncio

MODEL_PATH = r"C:\Users\atoca\Desktop\Naka-chan\kokoro-v1.0.onnx"
VOICES_PATH = r"C:\Users\atoca\Desktop\Naka-chan\voices-v1.0.bin"
ttts = TTTS(
    voice="af_heart",
    model_path=MODEL_PATH,
    voices_path=VOICES_PATH
)

recognizer = EnergyVADWhisper()

async def speech_loop():
    default_channel = None
    last = None
    while True:
        text = await recognizer.listen_realtime()
        print("recognized  🧑‍🎤 :" ,text)
        clean = text.strip()
        if clean and clean != last or clean != "thank you" or clean != "Thank you.":
            last = clean
            print("tttts bootin")
            asyncio.create_task(ttts.speak(text,speed = 0.85,pitch=1.2))
        else:
            print("⚠️  Dropped duplicate:", text)
        await asyncio.sleep(0.2)
if __name__ == "__main__":
    #features you use
    asyncio.run(speech_loop())
