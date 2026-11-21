"""Build scenario files and coverage curves.
  random set : plausible random driving scenes
  guided set : coverage-guided loop. After a small random start, an LLM is shown the UNCOVERED bins
               and proposes new scenes (JSON). Backends (env LLM_BACKEND):
                 rules  - built-in directed generator (no LLM, always works; use as fallback)
                 param  - BharatGen Param via Hugging Face transformers (local)
                 codex  - calls the Codex CLI non-interactively (`codex exec`); check `codex --help` on your install
                 manual - prompt is printed; you paste the reply from any LLM (Codex, Param, ...)
Usage: python gen_scenarios.py [N_TESTS]"""
import os, sys, json, re, subprocess
import numpy as np
import model as m

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BACKEND = os.environ.get("LLM_BACKEND", "rules")
MODEL_ID = os.environ.get("PARAM_MODEL", "bharatgenai/Param-1-2.9B-Instruct")
OUT = os.path.join(m.ROOT, "results"); os.makedirs(OUT, exist_ok=True)
Q = m.load()
rng = np.random.default_rng(7)

DEFAULT = {**{"d%d" % i: 20.0 for i in range(8)}, "speed": 30.0, "ped": 0, "wet": 0, "lane": 0.0}
KEYS = list(DEFAULT)

def clean(s):
    try:
        out = {k: float(s.get(k, DEFAULT[k])) for k in KEYS}
        for k in ("ped", "wet"): out[k] = int(out[k] > 0.5)
        return out
    except Exception:
        return None

def corner_scene():
    s = {"d%d" % i: float(rng.choice([0, 1, 15, 30])) for i in range(8)}
    s.update(speed=float(rng.choice([0, 60])), ped=int(rng.integers(0, 2)),
             wet=int(rng.integers(0, 2)), lane=float(rng.choice([-1, 0, 1])))
    return s

TEMPLATES = {
    "ped_close": dict(d0=10.0, ped=1), "front_close": dict(d0=6.0),
    "sensor_dropout": dict(d3=0.0, d5=0.0), "all_max_range": {"d%d" % i: 30.0 for i in range(8)},
    "conflict_lr": dict(d6=1.5, d2=1.5), "left_close": dict(d6=1.0), "right_close": dict(d2=1.0),
    "wet_fast": dict(wet=1, speed=58.0),
    "extreme_inputs": {**{"d%d" % i: 30.0 for i in range(8)}, "speed": 60.0, "ped": 1, "wet": 1, "lane": 1.0},
}

def maximize_acc():
    """Directed search: push every input to the extreme that raises the largest layer-1 accumulator."""
    s = dict(DEFAULT)
    opts = {**{"d%d" % i: [0.0, 30.0] for i in range(8)}, "speed": [0.0, 60.0], "ped": [0, 1], "wet": [0, 1], "lane": [-1.0, 1.0]}
    score = lambda t: int(np.abs(m.golden(Q, m.encode(t))["acc1"]).max())
    best = score(s)
    for _ in range(3):
        for k, vals in opts.items():
            for v in vals:
                t = {**s, k: v}; sc = score(t)
                if sc > best: s, best = t, sc
    return s

def rules_backend(uncovered):
    scenes = []
    for b in uncovered:
        if b in TEMPLATES:
            scenes.append({**DEFAULT, **TEMPLATES[b]})
        elif b.startswith("act_"):
            scenes += [random_until_action(b[4:])]
        elif b == "acc_large":
            scenes.append(maximize_acc())
        else:                                    # no template: try corner scenes
            scenes += [corner_scene() for _ in range(6)]
    return scenes

def random_until_action(name):
    for _ in range(300):
        s = m.random_scene(rng)
        if m.ACTIONS[m.golden(Q, m.encode(s))["action"]] == name: return s
    return m.random_scene(rng)

PROMPT = """You are a test engineer for a self-driving-car neural network chip.
A scene has keys: d0..d7 (obstacle distance in metres, 0 to 30, 0 means dead sensor; d0=front, d2=right, d6=left),
speed (0-60 km/h), ped (0 or 1), wet (0 or 1), lane (-1.0 to 1.0).
These coverage bins are NOT yet covered: {bins}.
Bin meanings: ped_close=front<12m with pedestrian; front_close=front<8m; sensor_dropout=a distance is 0;
all_max_range=all distances >=29.5; conflict_lr=left and right both <3m; left_close/right_close=<3m;
wet_fast=wet road and speed>50; extreme_inputs=every input at an extreme value;
hidden_large / hidden_negative / acc_large = internal values very large or very negative
(try extreme combinations of all inputs); act_X = chip outputs action X.
Reply with ONLY a JSON array of {n} scenes, each a JSON object with all 12 keys."""

_pipe = None
def llm_backend(uncovered, n=10):
    global _pipe
    prompt = PROMPT.format(bins=", ".join(uncovered), n=n)
    if BACKEND == "manual":
        print("\n----- COPY THIS PROMPT TO YOUR LLM -----\n" + prompt + "\n----- PASTE JSON REPLY, then an empty line -----")
        text = ""
        while True:
            line = sys.stdin.readline()
            if not line.strip(): break
            text += line
    elif BACKEND == "codex":
        r = subprocess.run(["codex", "exec", prompt], capture_output=True, text=True, timeout=600)
        text = r.stdout
    else:
        if _pipe is None:
            from transformers import pipeline
            _pipe = pipeline("text-generation", model=MODEL_ID, device_map="auto")
        text = _pipe(prompt, max_new_tokens=900, do_sample=True, temperature=0.7,
                     return_full_text=False)[0]["generated_text"]
    log.append({"prompt": prompt, "reply": text})
    mt = re.search(r"\[.*\]", text, re.S)
    try: arr = json.loads(mt.group(0)) if mt else []
    except Exception: arr = []
    return [c for c in (clean(s) for s in arr if isinstance(s, dict)) if c]

def curve(scenes):
    seen, pts = set(), []
    for s in scenes:
        seen |= m.hit_bins(Q, s); pts.append(len(seen))
    return pts, seen

log = []
random_set = [m.random_scene(rng) for _ in range(N)]
guided = [m.random_scene(rng) for _ in range(20)]            # same small random start
seen = set()
for s in guided: seen |= m.hit_bins(Q, s)
rounds = 0
while len(guided) < N:
    unc = [b for b in m.BINS if b not in seen]
    if not unc or rounds >= 30:
        guided += [m.random_scene(rng) for _ in range(N - len(guided))]; break
    new = (rules_backend(unc) if BACKEND == "rules" else llm_backend(unc))
    if not new: new = rules_backend(unc)                      # LLM gave nothing usable: fall back, keep going
    new = new[: N - len(guided)]
    guided += new; rounds += 1
    for s in new: seen |= m.hit_bins(Q, s)
guided = guided[:N]

pr, sr = curve(random_set); pg, sg = curve(guided)
json.dump(random_set, open(os.path.join(OUT, "scen_random.json"), "w"))
json.dump(guided, open(os.path.join(OUT, "scen_guided.json"), "w"))
json.dump(log, open(os.path.join(OUT, "llm_log.json"), "w"), indent=1)
json.dump({"random": pr, "guided": pg, "backend": BACKEND, "total_bins": len(m.BINS),
           "uncovered_random": [b for b in m.BINS if b not in sr],
           "uncovered_guided": [b for b in m.BINS if b not in sg]},
          open(os.path.join(OUT, "coverage_curves.json"), "w"), indent=1)
print("backend=%s | random: %d/%d bins | guided: %d/%d bins | LLM rounds: %d" %
      (BACKEND, pr[-1], len(m.BINS), pg[-1], len(m.BINS), rounds))
print("never hit by random:", [b for b in m.BINS if b not in sr])
print("never hit by guided:", [b for b in m.BINS if b not in sg])
