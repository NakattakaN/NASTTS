from na_tts import TTTS
import keyboard
from na_stt import FastWhisperRecognizer
import asyncio


ttts = TTTS()

recognizer = FastWhisperRecognizer()

async def speech_loop():
    default_channel = None
    while True:
        text = await recognizer.listen_push_to_talk("*")
        if text.strip():
            await ttts.speak(text, pitch = "+20Hz")
        await asyncio.sleep(0.1)
if __name__ == "__main__":
    #features you use
    discordd = False
    screen = False
    voice = True
    talk_local = True
    asyncio.run(speech_loop())
