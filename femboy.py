from na_tts import TTTS
import keyboard
from na_stt import EnergyVADWhisper
import asyncio


ttts = TTTS()

recognizer = EnergyVADWhisper()

async def speech_loop():
    default_channel = None
    last = None
    while True:
        text = await recognizer.listen_realtime()
        clean = text.strip()
        if clean and clean != last or clean != "thank you" or clean != "Thank you":
            last = clean
            asyncio.create_task(ttts.speak(text, pitch="+35Hz",rate ="-10%"))
        else:
            print("⚠️  Dropped duplicate:", text)
        await asyncio.sleep(0.2)
if __name__ == "__main__":
    #features you use
    asyncio.run(speech_loop())
