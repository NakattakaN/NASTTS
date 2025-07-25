import asyncio
import edge_tts
import sys
import shutil
import subprocess

class TTTS:
    def __init__(self, voice="en-AU-NatashaNeural"):
        self.voice = voice
        if not shutil.which("ffplay"):
            raise RuntimeError("ffplay not found in PATH—please install ffmpeg.")

    async def speak(self, text: str, pitch: str = "+20Hz", rate: str = "+10%"):
        # Create ffplay process with stdin pipe
        ffplay = await asyncio.create_subprocess_exec(
            "ffplay", 
            "-nodisp", 
            "-autoexit", 
            "-loglevel", "quiet", 
            "-",  # Read from stdin
            stdin=asyncio.subprocess.PIPE
        )

        # Stream audio directly to ffplay
        communicate = edge_tts.Communicate(text, self.voice, pitch=pitch, rate=rate)
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Write audio chunks to ffplay's stdin
                    ffplay.stdin.write(chunk["data"])
            await ffplay.stdin.drain()
        finally:
            # Ensure process termination
            ffplay.stdin.close()
            await ffplay.wait()

async def main():
    tts = TTTS("en-GB-SoniaNeural")
    await tts.speak(
        "Monica Everett, a genius cunt, was extremely shy and terrible at speaking in public...",
        pitch="+35Hz", 
        rate="-10%"
    )

if __name__ == "__main__":
    asyncio.run(main())

