"""Whisper speech recognition + command parsing. Needs: pip install openai-whisper sounddevice"""
import re
_model = None

def parse_command(text):
    """Returns 'STOP', 'RESUME', 'TURN_LEFT', 'TURN_RIGHT' or None."""
    t = re.sub(r"[^a-z ]", " ", text.lower())
    if re.search(r"\b(emergency|stop|halt|brake)\b", t): return "STOP"
    if re.search(r"\b(resume|go|continue|start|drive)\b", t): return "RESUME"
    if re.search(r"\bleft\b", t): return "TURN_LEFT"
    if re.search(r"\bright\b", t): return "TURN_RIGHT"
    return None

def listen(seconds=3.0, model_name="base"):
    """Record from the microphone and return Whisper's transcript."""
    global _model
    import numpy as np, sounddevice as sd, whisper
    if _model is None: _model = whisper.load_model(model_name)
    audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    return _model.transcribe(np.squeeze(audio), fp16=False, language="en")["text"].strip()
