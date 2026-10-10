import sys
from pathlib import Path

def play_audio(file_path: str = "outputs/conversation.wav"):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found at '{path.resolve()}'")
        print("Please run 'python main.py' first to generate a conversation.")
        return

    print(f"Playing audio: {path.resolve()}")
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path.resolve()), winsound.SND_FILENAME)
        print("Playback finished.")
    else:
        import os
        os.system(f"aplay {path.resolve()} 2>/dev/null || afplay {path.resolve()} 2>/dev/null")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs/conversation.wav"
    play_audio(target)
