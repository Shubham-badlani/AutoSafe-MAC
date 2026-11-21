"""Verify the BharatGen integration: load Param, ask for one driving scene, parse it. Prints PASS/FAIL.
Needs: pip install transformers torch accelerate   (first run downloads the model)
Override the model with PARAM_MODEL=<hugging-face-id>; read the model card for any extra loading flags."""
import os, json, re
MODEL_ID = os.environ.get("PARAM_MODEL", "bharatgenai/Param-1-2.9B-Instruct")
from transformers import pipeline
pipe = pipeline("text-generation", model=MODEL_ID, device_map="auto")
prompt = ('Reply with ONLY a JSON array containing 2 driving scenes. Each scene is an object with keys '
          'd0,d1,d2,d3,d4,d5,d6,d7 (obstacle distance in metres, 0-30), speed (0-60), ped (0 or 1), wet (0 or 1), lane (-1 to 1). '
          'Make one scene with a pedestrian very close in front, and one with obstacles on both left and right.')
out = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, return_full_text=False)[0]["generated_text"]
print("RAW REPLY:\n", out)
mt = re.search(r"\[.*\]", out, re.S)
try:
    arr = json.loads(mt.group(0)); ok = isinstance(arr, list) and len(arr) > 0 and all("d0" in a for a in arr)
except Exception:
    ok = False
print("BHARATGEN INTEGRATION:", "PASS" if ok else "FAIL (see raw reply; try manual backend or a clearer prompt)")
