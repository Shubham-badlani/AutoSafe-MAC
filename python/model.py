"""AutoSafe-MAC: dataset, training, INT8 quantization, golden model, coverage bins.
Pure NumPy. The golden model is bit-exact with rtl/autosafe_mac.v."""
import os
import numpy as np

ACTIONS = ["GO", "STOP", "TURN_LEFT", "TURN_RIGHT"]
N_IN, N_HID, N_OUT = 12, 16, 4
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEIGHTS = os.path.join(HERE, "weights.npz")

# scene keys: d0..d7 (metres 0-30; 0 = dead sensor), speed (0-60), ped (0/1), wet (0/1), lane (-1..1)
# d0 = front, d2 = right, d6 = left
def encode(s):
    d = [min(max(float(s["d%d" % i]), 0), 30) for i in range(8)]
    x = [round(v * 4) for v in d]                                  # 0..120
    x.append(round(min(max(float(s["speed"]), 0), 60) * 2))        # 0..120
    x.append(100 if int(s["ped"]) else 0)
    x.append(100 if int(s["wet"]) else 0)
    x.append(round(min(max(float(s["lane"]), -1), 1) * 100))       # -100..100
    return np.array(x, dtype=np.int64)

def rule_label(s):
    """Ground-truth driving rule (the 'safety spec'). 0 GO, 1 STOP, 2 TURN_LEFT, 3 TURN_RIGHT."""
    front, left, right = float(s["d0"]), float(s["d6"]), float(s["d2"])
    if front < 8 or (int(s["ped"]) and front < 12): return 1      # STOP
    if left < 3: return 3                                         # obstacle on the left -> turn right
    if right < 3: return 2                                        # obstacle on the right -> turn left
    return 0

def random_scene(rng):
    s = {"d%d" % i: float(rng.uniform(1, 30)) for i in range(8)}
    s.update(speed=float(rng.uniform(0, 60)), ped=int(rng.integers(0, 2)),
             wet=int(rng.integers(0, 2)), lane=float(rng.uniform(-1, 1)))
    return s

def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    scenes = [random_scene(rng) for _ in range(n)]
    X = np.array([encode(s) for s in scenes])
    y = np.array([rule_label(s) for s in scenes])
    return scenes, X, y

def train(seed=0, epochs=3000, lr=0.02, wd=1e-3):
    _, X, y = make_dataset(6000, seed)
    Xf = X / 127.0
    rng = np.random.default_rng(seed + 1)
    P = [rng.normal(0, 0.5, (N_HID, N_IN)), np.zeros(N_HID),
         rng.normal(0, 0.5, (N_OUT, N_HID)), np.zeros(N_OUT)]
    Y = np.eye(N_OUT)[y]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    for t in range(1, epochs + 1):
        W1, b1, W2, b2 = P
        h = np.maximum(Xf @ W1.T + b1, 0)
        z = h @ W2.T + b2
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        dz = (p - Y) / len(Xf); dh = dz @ W2; dh[h <= 0] = 0
        G = [dh.T @ Xf + wd * W1, dh.sum(0), dz.T @ h + wd * W2, dz.sum(0)]
        for i in range(4):                                   # Adam
            M[i] = 0.9 * M[i] + 0.1 * G[i]; V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
            P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
    return tuple(P)

def float_predict(P, X):
    W1, b1, W2, b2 = P
    h = np.maximum((X / 127.0) @ W1.T + b1, 0)
    return (h @ W2.T + b2).argmax(1)

def quantize(P, Xcal=None):
    """INT8 weights; the layer-1 shift is calibrated so hidden activations use the INT8 range."""
    W1, b1, W2, b2 = P
    if Xcal is None:
        _, Xcal, _ = make_dataset(3000, 123)
    sw1 = 127.0 / np.abs(W1).max()
    w1q = np.clip(np.round(W1 * sw1), -128, 127).astype(np.int64)
    b1q = np.round(b1 * sw1 * 127).astype(np.int64)
    pre = Xcal @ w1q.T + b1q
    top = np.percentile(np.maximum(pre, 1), 99.5)
    sh1 = max(0, int(np.ceil(np.log2(top / 127.0))))
    h_scale = 127.0 * sw1 / (2 ** sh1)          # hidden INT8 value = h_real * h_scale
    sw2 = 127.0 / np.abs(W2).max()
    w2q = np.clip(np.round(W2 * sw2), -128, 127).astype(np.int64)
    b2q = np.round(b2 * sw2 * h_scale).astype(np.int64)
    return dict(w1=w1q, b1=b1q, w2=w2q, b2=b2q, sh1=np.array([sh1]))

def save(Q):
    np.savez(WEIGHTS, **Q)

def load():
    z = np.load(WEIGHTS)
    return {k: z[k] for k in z.files}

def golden(Q, x):
    """x: 12 INT8 values. Bit-exact model of the hardware."""
    acc1 = Q["w1"] @ x
    pre = (acc1 + Q["b1"]) >> int(Q["sh1"][0])           # arithmetic shift (floor)
    h = np.minimum(np.maximum(pre, 0), 127)
    logits = Q["w2"] @ h + Q["b2"]
    return dict(action=int(np.argmax(logits)), logits=logits, acc1=acc1, pre=pre, h=h)

BINS = ["ped_close", "front_close", "sensor_dropout", "all_max_range", "conflict_lr",
        "left_close", "right_close", "wet_fast", "extreme_inputs",
        "hidden_large", "hidden_negative", "acc_large",
        "act_GO", "act_STOP", "act_TURN_LEFT", "act_TURN_RIGHT"]

def hit_bins(Q, s):
    x = encode(s); g = golden(Q, x)
    d = [float(s["d%d" % i]) for i in range(8)]
    hit = set()
    if d[0] < 12 and int(s["ped"]): hit.add("ped_close")
    if d[0] < 8: hit.add("front_close")
    if any(v <= 0.01 for v in d): hit.add("sensor_dropout")
    if all(v >= 29.5 for v in d): hit.add("all_max_range")
    if d[6] < 3 and d[2] < 3: hit.add("conflict_lr")
    if d[6] < 3: hit.add("left_close")
    if d[2] < 3: hit.add("right_close")
    if int(s["wet"]) and float(s["speed"]) > 50: hit.add("wet_fast")
    if np.abs(x).min() >= 100: hit.add("extreme_inputs")
    if (g["pre"] > 63).any(): hit.add("hidden_large")
    if (g["pre"] < 0).any(): hit.add("hidden_negative")
    if (np.abs(g["acc1"]) > 14500).any(): hit.add("acc_large")
    hit.add("act_" + ACTIONS[g["action"]])
    return hit
